
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

import Data.List (foldl')
import Data.Array.IArray
import qualified Data.Packed.Matrix as Mt
import qualified Data.Packed.Vector as Mt
import qualified Numeric.LinearAlgebra as Mt
import qualified Numeric.LinearAlgebra.Algorithms as Mt
import Data.IntMap.Strict (IntMap)
import qualified Data.IntMap.Strict as IM

import qualified Data.ByteString as B
import qualified Data.ByteString.UTF8 as UTF8
import qualified Data.ByteString.Char8 as C8
import qualified Vocab as Vocab
import qualified HSNeuralNet as NN
import Util
import PCA (pca)

type DVec = Mt.Vector Double
type DMat = Mt.Matrix Double

runAllWords :: Vocab.Vocab -> B.ByteString -> Int -> IO ()
runAllWords vocab content dimens = do
	net  <- NN.randomNetwork (Vocab.uniqueWords vocab) dimens
	net_ <- iter vocab net 0
	putStrLn $ "complete: "
	--putStrLn $ "feature: " ++ (show $ IM.toList ft_)
	--putStrLn $ "output:  " ++ (show $ ot_)
	-- let v = NN.runNetwork net 2
	--let outs = Mt.mapVectorWithIndex (\i _ -> softmax v i) v
	--putStrLn $ "full softmax output: " ++ (show $ map (printf "%.2f" :: Double->String) $ Mt.toList outs)
	let features = NN.featureVecArray net_
	a <- plot $ imap (\i x -> (UTF8.toString $ Vocab.findIndex vocab i, x Mt.@> 0, x Mt.@> 1) ) $ pca 2 features
	return ()
	where iter vocab net x = do
		net2 <- evalRandIO $ Vocab.doIteration vocab content 5 (0.025, 0.0001) NN.runWord net
		putStrLn $ "iteration " ++ (show x) ++ " complete "  ++ (show $ NN.forceEval net2)
		--let v = runNetwork (ft2 IM.! 0) ot2
		--et outs = Mt.mapVectorWithIndex (\i _ -> softmax v i) v
		--putStrLn $ "full softmax output: " ++ (show $ map (printf "%.2f" :: Double->String) $ Mt.toList outs)
		--putStrLn $ "network output     : " ++ (show $ map (printf "%.2f" :: Double->String) $ Mt.toList v)
		if x < 0 then iter vocab net2 (x + 1) else return net2

main = do
	crps <- B.readFile "corpus.txt"
	let vocab = Vocab.makeVocab (Vocab.countWordFreqs $ C8.words crps) 5
	putStrLn $ "Vocab loading complete " ++ (show $ Vocab.uniqueWords vocab)
	runAllWords vocab crps 100
	return ()

