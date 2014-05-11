
-- let's use skipgrams
-- the basic approach:
-- each word in the vocabulary is an input node
-- the projection layer has 200 or so nodes
-- each word in the vocabulary is an output node, too
-- for each word in the vocabulary:
   -- for each occurence:
	  -- select a random value R, and look up R
	  --   words before and R after the occurence
	  -- add them to the training data
 -- shuffle the training data
 -- now train the neural net, checking the output with a softmax function
 -- (the input is a word, the output should be a word from the context)
 -- do this until the log-likelihood is high enough / doesn't get higher

-- another way to see this: each word has a feature vector,
--  multiply this with the output weight matrix to get the output values
-- softmax it based on the output word, and backprop it

import System.Random
import Control.Monad (foldM, liftM, replicateM)
import Control.Monad.Random (Rand, evalRandIO)
import Debug.Trace
import Text.Printf
import Data.String.Utils
import Control.DeepSeq

import Data.List (foldl')
import qualified Data.Packed.Matrix as Mt
import qualified Data.Packed.Vector as Mt
import qualified Numeric.LinearAlgebra as Mt
import qualified Numeric.LinearAlgebra.Algorithms as Mt
import Data.IntMap.Strict (IntMap)
import qualified Data.IntMap.Strict as IM
import System.Cmd (rawSystem)
import System.Exit (ExitCode (ExitSuccess))
import qualified Data.ByteString.UTF8 as UTF8
import qualified Corpus as Corpus

type DVec = Mt.Vector Double
type DMat = Mt.Matrix Double
-- output is a V * feat matrix
runNetwork :: DVec -> DMat -> DVec
runNetwork feature output = feature `Mt.vXm` output

-- exp(a_i) / sum(k)(exp(a_k))
softmax :: DVec -> Int -> Double
softmax a i = (expa Mt.@> i) / (Mt.sumElements expa)
	where expa = Mt.mapVector exp $ a
-- derivative over q_k: = (d(k,i) - s(a,k)) * s(a,i)
-- k and i can be swapped
-- from wikipedia/Softmax_function
softmax' :: DVec -> Int -> Int -> Double
softmax' a i k = ((delta k i) - softmax a k) * softmax a i
	where delta a b = if a == b then 1 else 0
logsoftmax :: DVec -> Int -> Double
logsoftmax a i = if (0 == softmax a i) then error "log of zero" else log $ softmax a i
-- derivative over q_k: s'(a, i, k) / s(a, i)
-- k and i can *not* be swapped
logsoftmax' :: DVec -> Int -> Int -> Double
logsoftmax' a i k = (softmax' a i k) / (softmax a i)

imap' :: (Int -> a -> b) -> Int -> [a] -> [b]
imap' _ _ []     = []
imap' f i (x:xs) = f i x : imap' f (i + 1) xs
imap = flip imap' 0

runWord :: DVec -> DMat -> Int -> (DVec, DMat)
runWord feature output expected = (newfeat, newout)
	where
		out = runNetwork feature output
		softerr = 0 - (logsoftmax out expected)
		-- calculate the delta for the output layer
		-- the softmax can be seen as another layer on top of it with fixed weight 1
		-- this means this is also the modified error of the output layer
		-- delta_k = Err_k * g'(in_k)
		moderr = Mt.mapVector (* softerr) $ Mt.buildVector (Mt.dim out) (logsoftmax' out expected)
		-- calculate the weight adjustment between the output and projection
		-- W_(j,k) := W_(j,k) + alpha * a_j * delta_k
		newout = Mt.fromColumns $ imap processout $ Mt.toColumns output
		processout k feat = Mt.zipVectorWith (adjustprojoutcon (moderr Mt.@> k)) feature feat
		adjustprojoutcon d_k activation weight = weight + rate * d_k * activation
		-- calculate the propagation deltas for the projection layer
		delta_j = output `Mt.mXv` moderr
		-- calculate the new weights for the feature vector
		newfeat = check_size feature delta_j $ Mt.zipVectorWith (\w d -> w + rate * d) feature delta_j
		--output !! k ! j + rate * (feature ! j) * (moderr ! k)
		rate = 0.005 :: Double
		traceShowId a = trace (show a) a
		check_size a b x = if (Mt.dim a /= Mt.dim b) then error "size doesn't match" else x


randomNetwork :: Int -> Int -> IO (DMat, DMat)
randomNetwork vocab dimen = do
	seeda <- randomIO :: IO Int
	seedb <- randomIO :: IO Int
	let a = Mt.randomVector seeda Mt.Uniform (vocab * dimen)
	let b = Mt.randomVector seedb Mt.Uniform (vocab * dimen)
	return (Mt.reshape dimen a, Mt.reshape vocab b)


runWords :: (RandomGen g) => Corpus.Corpus -> IntMap DVec -> DMat -> Rand g (IntMap DVec, DMat)
runWords words feat out = do
	pairs <- Corpus.wordPairs words
	return $! foldl' runWordPair (feat, out) pairs
	where
		runWordPair (f, o) (a, b) = (IM.insert a thisf f, newo)
			where (thisf, newo) = runWord (f IM.! a) o b

runAllWords :: Corpus.Corpus -> Int -> IO ()
runAllWords wrds dimens = do
	(feat, out) <- randomNetwork (Corpus.numWords wrds) dimens
	let ft = IM.fromAscList $ imap (,) (Mt.toRows feat)
	(ft_, ot_) <- iter (ft, out) 0
	putStrLn $ "complete: "
	--putStrLn $ "feature: " ++ (show $ IM.toList ft_)
	--putStrLn $ "output:  " ++ (show $ ot_)
	let v = runNetwork (ft_ IM.! 2) ot_
	let outs = Mt.mapVectorWithIndex (\i _ -> softmax v i) v
	--putStrLn $ "full softmax output: " ++ (show $ map (printf "%.2f" :: Double->String) $ Mt.toList outs)
	a <- plot $ imap (\i x -> (UTF8.toString $ Corpus.findIndex wrds i, x Mt.@> 0, x Mt.@> 1) ) $ pca 2 ft_
	return ()
	where iter (ft, ot) x = do
		(ft2, ot2) <- evalRandIO $ runWords wrds ft ot
		putStrLn $ "iteration " ++ (show x) ++ " complete " ++ (show $ IM.size $ ft2)
		--let v = runNetwork (ft2 IM.! 0) ot2
		--et outs = Mt.mapVectorWithIndex (\i _ -> softmax v i) v
		--putStrLn $ "full softmax output: " ++ (show $ map (printf "%.2f" :: Double->String) $ Mt.toList outs)
		--putStrLn $ "network output     : " ++ (show $ map (printf "%.2f" :: Double->String) $ Mt.toList v)
		if x < 2 then iter (ft2, ot2) (x + 1) else return (ft2, ot2)

main = do
	crps <- Corpus.getFullCorpus
	putStrLn $ "corpus loading complete " ++ (show $ Corpus.numWords crps)
	a <- evalRandIO $ Corpus.wordPairs crps
	runAllWords crps 110
	return ()

normalize_mean :: [Mt.Vector Double] -> [Mt.Vector Double]
normalize_mean x = map (flip (-) mean) x where
	mean = (sum x) / (fromIntegral $ length x)

covariance_matrix :: [Mt.Vector Double] -> DMat
covariance_matrix x = (sum $ map (\a -> (Mt.asColumn a) * (Mt.asRow a)) x) / (fromIntegral $ length x)

eigenvectors :: Mt.Matrix Double -> [Mt.Vector Double]
eigenvectors x = Mt.toColumns x where
	(u, s, v) = Mt.svd x

pca :: Int -> IntMap DVec -> [DVec]
pca dims x = map (`Mt.vXm` base) dataset
	where
		base = Mt.fromColumns $ take dims $ eigenvectors $ covariance_matrix $ dataset
		dataset = normalize_mean $ toVecArray x
		toVecArray :: IntMap DVec -> [DVec]
		toVecArray x = map snd $ IM.toAscList x

-- loosely based on http://hackage.haskell.org/package/easyplot-1.0/docs/src/Graphics-EasyPlot.html#Plot
plot :: (Show a, Num a) => [(String, a, a)] -> IO Bool
plot points = do
	writeFile filename dataset
	exitCode <- rawSystem "gnuplot" args
	return $ exitCode == ExitSuccess
	where 
		-- todo see if this works when haskell uses scientific notation
		dataset = unlines $ map (\(s, a, b) -> s ++ " " ++ (show a) ++ " " ++ (show b)) points
		args = ["-e", join ";" [
				"set term png size 1024,1024",
				--"set offsets 1,1,1,1",
				"set output \"pca.png\"",
				"plot \"" ++ filename ++ "\" using 2:3:1 with labels title \"\""]]
		filename = "plot1.dat"
