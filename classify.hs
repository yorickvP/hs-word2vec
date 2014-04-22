
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
-- XXX: derivations of softmax have a k parameter, what is this for?

import System.Random
import System.Random.Shuffle
import Control.Monad (forM_, foldM, liftM, replicateM)
import Control.Monad.Random
import Debug.Trace
import Data.List (foldl)
import Data.Foldable (concatMap, fold)
import Text.Printf
import Data.String.Utils
import qualified Data.Packed.Matrix as Mt
import qualified Data.Packed.Vector as Mt
import qualified Numeric.LinearAlgebra as Mt
import qualified Numeric.LinearAlgebra.Algorithms as Mt
import qualified Data.Traversable as T
import qualified Data.Map as M
import System.Cmd (rawSystem)
import System.Exit (ExitCode (ExitSuccess))

type DVec = Mt.Vector Double
type DMat = Mt.Matrix Double
-- output is a V * feat matrix
runNetwork :: DVec -> DMat -> DVec
runNetwork feature output = feature `Mt.vXm` output

-- exp(a_i) / sum(k)(exp(a_k))
softmax :: DVec -> Int -> Double
softmax a i = (expa Mt.@> i) / (Mt.sumElements expa)
	where expa = Mt.mapVector exp $ a
-- derivative: (d(k,i) - s(a,k)) * s(a,i)
-- k and i can be swapped, so I don't have to figure out
-- which does what
softmax' :: DVec -> Int -> Int -> Double
softmax' a i k = ((delta k i) - softmax a k) * softmax a i
	where delta a b = if a == b then 1 else 0
logsoftmax :: DVec -> Int -> Double
logsoftmax a i = if (0 == softmax a i) then error "log of zero" else log $ softmax a i
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
		-- calculate the delta-k for the output layer
		-- the softmax can be seen as another layer on top of it with fixed weight 1
		-- this means this is also the modified error of the output layer
		moderr = Mt.mapVector (* softerr) $ Mt.buildVector (Mt.dim out) (logsoftmax' out expected)
		-- calculate the weight adjustment between the output and projection
		newout = Mt.fromColumns $ imap processout $ Mt.toColumns output
		processout k feat = Mt.zipVectorWith (adjustprojoutcon (moderr Mt.@> k)) feature feat
		adjustprojoutcon d_k activation weight = weight + rate * d_k * activation
		-- calculate the propagation deltas for the projection layer
		delta_j = Mt.zipVectorWith (*) (Mt.fromList $ map Mt.sumElements $ Mt.toRows output) moderr
		-- calculate the new weights for the feature vector
		newfeat = Mt.zipVectorWith (\w d -> w + rate * d) feature delta_j
		--output !! k ! j + rate * (feature ! j) * (moderr ! k)
		rate = 0.005 :: Double
		traceShowId a = trace (show a) a


randomNetwork :: Int -> Int -> IO (DMat, DMat)
randomNetwork vocab dimen = do
	alist <- replicateM (vocab * dimen) randomIO
	blist <- replicateM (vocab * dimen) randomIO
	return ((vocab Mt.>< dimen)alist, (dimen Mt.>< vocab)blist)

wordStuff = M.fromList [
		("the",   (0, [([], ["quick", "brown", "fox"]), (["fox", "jumps", "over"], ["lazy", "dog"])])),
		("quick", (1, [(["the"], ["brown", "fox", "jumps"])])),
		("brown", (2, [(["the", "quick"], ["fox", "jumps", "over"])])),
		("fox",   (3, [(["the", "quick", "brown"], ["jumps", "over", "the"])])),
		("jumps", (4, [(["quick", "brown", "fox"], ["over", "the", "lazy"])])),
		("over",  (5, [(["brown", "fox", "jumps"], ["the", "lazy", "dog"])])),
		("lazy",  (6, [(["jumps", "over", "the"], ["dog"])])),
		("dog",   (7, [(["over", "the", "lazy"], [])]))
	]


-- from stackoverflow
lastN :: Int -> [a] -> [a]
lastN n xs = foldl (const .drop 1) xs (drop n xs)

-- convert a word list like above into shuffled pairs of training data
wordPairs :: (RandomGen g) => M.Map String (Int, [([String], [String])]) -> Int -> Rand g [(Int, Int)]
wordPairs words c = 
	concatMapM indivWordPairs words >>= shuffleM
	where
		indivWordPairs (i, inst)        = concatMapM (instWordPairs i) inst
		instWordPairs i (before, after) = do
			r <- getRandomR (1, c)
			return $ map ((,) i . fst . (words M.!)) $ (lastN r before) ++ take r after
		concatMapM f = liftM fold . T.mapM f

runWords :: (RandomGen g) => M.Map String (Int, [([String], [String])]) -> Int -> M.Map Int DVec -> DMat -> Rand g (M.Map Int DVec, DMat)
runWords words c feat out = do
	pairs <- wordPairs words c
	foldM runWordPair (feat, out) pairs
	where
		runWordPair (f, o) (a, b) = do
			let (thisf, newo) = runWord (f M.! a) o b
			return (M.insert a thisf f, newo)

runAllWords :: IO ()
runAllWords = do
	(feat, out) <- randomNetwork 8 7
	let ft = M.fromAscList $ imap (,) (Mt.toRows feat)
	(ft_, ot_) <- iter (ft, out) 0
	putStrLn $ "complete: "
	putStrLn $ "feature: " ++ (show $ M.toList ft_)
	putStrLn $ "output:  " ++ (show $ ot_)
	let v = runNetwork (ft_ M.! 0) ot_
	let outs = Mt.mapVectorWithIndex (\i _ -> softmax v i) v
	putStrLn $ "full softmax output: " ++ (show $ map (printf "%.2f" :: Double->String) $ Mt.toList outs)
	let wordmap = M.fromList $ map (\(w, (x, _)) -> (x, w)) (M.toList wordStuff)
	a <- plot $ imap (\i x -> (wordmap M.! i, x Mt.@> 0, x Mt.@> 1) ) $ pca 2 ft_
	return ()
	where iter (ft, ot) x = do
		(ft2, ot2) <- evalRandIO $ runWords wordStuff 3 ft ot
		putStrLn $ "iteration " ++ (show x) ++ " complete " ++ (show $ sum $ map Mt.sumElements $ map snd $ M.toList $ ft2)
		let v = runNetwork (ft2 M.! 0) ot2
		let outs = Mt.mapVectorWithIndex (\i _ -> softmax v i) v
		putStrLn $ "full softmax output: " ++ (show $ map (printf "%.2f" :: Double->String) $ Mt.toList outs)
		putStrLn $ "network output     : " ++ (show $ map (printf "%.2f" :: Double->String) $ Mt.toList v)
		if x < 1000 then iter (ft2, ot2) (x + 1) else return (ft2, ot2)

main = runAllWords

normalize_mean :: [Mt.Vector Double] -> [Mt.Vector Double]
normalize_mean x = map (flip (-) mean) x where
	mean = (sum x) / (fromIntegral $ length x)

covariance_matrix :: [Mt.Vector Double] -> DMat
covariance_matrix x = (sum $ map (\a -> (Mt.asColumn a) * (Mt.asRow a)) x) / (fromIntegral $ length x)

eigenvectors :: Mt.Matrix Double -> [Mt.Vector Double]
eigenvectors x = Mt.toColumns x where
	(u, s, v) = Mt.svd x

pca :: Int -> M.Map Int DVec -> [DVec]
pca dims x = map (`Mt.vXm` base) dataset
	where
		base = Mt.fromColumns $ take dims $ eigenvectors $ covariance_matrix $ dataset
		dataset = normalize_mean $ toVecArray x
		toVecArray :: M.Map Int DVec -> [DVec]
		toVecArray x = map snd $ M.toAscList x

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
				"set term png",
				"set offsets 1,1,1,1",
				"set output \"pca.png\"",
				"plot \"" ++ filename ++ "\" using 2:3:1 with labels title \"\""]]
		filename = "plot1.dat"
