
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

import Data.List (foldl')
import Data.Array.IArray
import qualified Data.Packed.Matrix as Mt
import qualified Data.Packed.Vector as Mt
import qualified Numeric.LinearAlgebra as Mt
import qualified Numeric.LinearAlgebra.Algorithms as Mt
import Data.IntMap.Strict (IntMap)
import qualified Data.IntMap.Strict as IM
import System.Process (rawSystem)
import System.Exit (ExitCode (ExitSuccess))
import qualified Data.ByteString as B
import qualified Data.ByteString.UTF8 as UTF8
import qualified Data.ByteString.Char8 as C8
import qualified Vocab as Vocab
import qualified NeuralNet as NN
import Util

type DVec = Mt.Vector Double
type DMat = Mt.Matrix Double

runAllWords :: Vocab.Vocab -> B.ByteString -> Int -> IO ()
runAllWords vocab content dimens = do
	net  <- NN.randomNetwork (Vocab.uniqueWords vocab) dimens
	net_ <- iter vocab net 0
	putStrLn $ "complete: "
	--putStrLn $ "feature: " ++ (show $ IM.toList ft_)
	--putStrLn $ "output:  " ++ (show $ ot_)
	let v = NN.runNetwork net 2
	--let outs = Mt.mapVectorWithIndex (\i _ -> softmax v i) v
	--putStrLn $ "full softmax output: " ++ (show $ map (printf "%.2f" :: Double->String) $ Mt.toList outs)
	let features = NN.featureVecArray net_
	a <- plot $ imap (\i x -> (UTF8.toString $ Vocab.findIndex vocab i, x Mt.@> 0, x Mt.@> 1) ) $ pca 2 features
	return ()
	where iter vocab net x = do
		net2 <- evalRandIO $ Vocab.doIteration vocab content 5 NN.runWord net
		putStrLn $ "iteration " ++ (show x) ++ " complete "  ++ (show $ NN.forceEval net2)
		--let v = runNetwork (ft2 IM.! 0) ot2
		--et outs = Mt.mapVectorWithIndex (\i _ -> softmax v i) v
		--putStrLn $ "full softmax output: " ++ (show $ map (printf "%.2f" :: Double->String) $ Mt.toList outs)
		--putStrLn $ "network output     : " ++ (show $ map (printf "%.2f" :: Double->String) $ Mt.toList v)
		if x < 10 then iter vocab net2 (x + 1) else return net2

main = do
	crps <- B.readFile "corpus.txt"
	let vocab = Vocab.makeVocab (Vocab.countWordFreqs $ C8.words crps) 2
	putStrLn $ "Vocab loading complete " ++ (show $ Vocab.uniqueWords vocab)
	runAllWords vocab crps 150
	return ()

normalize_mean :: [Mt.Vector Double] -> [Mt.Vector Double]
normalize_mean x = map (flip (-) mean) x where
	mean = (sum x) / (fromIntegral $ length x)

covariance_matrix :: [Mt.Vector Double] -> DMat
covariance_matrix x = (sum $ map (\a -> (Mt.asColumn a) * (Mt.asRow a)) x) / (fromIntegral $ length x)

eigenvectors :: Mt.Matrix Double -> [Mt.Vector Double]
eigenvectors x = Mt.toColumns x where
	(u, s, v) = Mt.svd x

pca :: Int -> [DVec] -> [DVec]
pca dims x = map (`Mt.vXm` base) dataset
	where
		base = Mt.fromColumns $ take dims $ eigenvectors $ covariance_matrix $ dataset
		dataset = normalize_mean $ x

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
