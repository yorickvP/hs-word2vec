
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

import Control.Monad.Random (evalRandIO)

import Data.List (intersperse)
import qualified Data.Packed.Vector as Mt

import qualified Data.ByteString as B
import qualified Data.ByteString.Char8 as C8
import qualified Data.ByteString.Lazy as L
import qualified Vocab as Vocab
import qualified HSNeuralNet as NN


runAllWords :: Vocab.Vocab -> B.ByteString -> Int -> IO ()
runAllWords vocab content dimens = do
	net  <- NN.randomNetwork (Vocab.uniqueWords vocab) dimens
	net_ <- wordsIteration net
	putStrLn "training complete, writing to outwords.txt"
	let sorted_vocab = Vocab.sortedVecList vocab (NN.getFeat net_)
	L.writeFile "outwords.txt" (
		-- bytestring unlines
		L.fromChunks $ intersperse (C8.pack "\n") $
		-- first line: number of words + number of dimensions
		(C8.pack $ unwords $ map show [Vocab.uniqueWords vocab,dimens]) : 
		-- other lines: word vec vec vec vec
		map (C8.unwords . (\(b, vecs) -> b : (map (C8.pack . show) $ Mt.toList vecs))) sorted_vocab)
	return ()
	where
		wordsIteration :: NN.NeuralNet -> IO NN.NeuralNet
		wordsIteration net = do
			let itercount = 1 :: Int
			-- fold NN.runWord over all the training pairs
			-- max lookaround: 5, maxrate: 0.025, minrate: 0.0001
			net2 <- evalRandIO $ Vocab.doIteration vocab content 5 NN.runWord net
			putStrLn $ "iteration " ++ (show itercount) ++ " complete "  ++ (show $ NN.forceEval net2)
			return net2
			-- possibly run this multiple times, not needed
			--if itercount < 10 then wordsIteration vocab net2 (itercount + 1) else return net2

main :: IO ()
main = do
	crps <- B.readFile "corpus.txt"
	-- only use words that occur more than 5 times
	let vocab = Vocab.makeVocab (Vocab.countWordFreqs $
		concatMap (++ [C8.pack "</s>"]) $ map C8.words $ C8.lines crps) 5
	putStrLn $ "Vocab loading complete: " ++
		(show $ Vocab.wordCount vocab) ++ " total words, "
		++ (show $ Vocab.uniqueWords vocab) ++ " unique words"
	-- 100-dimensional vectors
	runAllWords vocab crps 100

