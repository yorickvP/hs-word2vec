
import Control.Monad.Random (evalRandT)
import Control.Monad.Writer
import System.Random (getStdGen)

import Data.List (intersperse)
import qualified Data.Packed.Vector as Mt

import qualified Data.ByteString as B
import qualified Data.ByteString.Char8 as C8
import qualified Data.ByteString.Lazy as L
import qualified Vocab as Vocab
import qualified HSNeuralNet as NN
import Util


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
			gen <- getStdGen
			let itercount = 1 :: Int
			-- fold NN.runWord over all the training pairs
			-- max lookaround: 5
			let func = Vocab.doIteration vocab content 5 NN.runWord net
			let (net2, statusupdates) = runWriter $ evalRandT func gen
			-- report on the progress from the writer monad (statusupdates is a lazy list)
			forM_ statusupdates (\(_rate, Vocab.TrainProgress itcount total, avg) ->
				putStrLn $ "iteration " ++ (show itcount) ++
								" / "   ++ (show total) ++
								" average: " ++ (show $ NN.calcAvg avg)
				)
			-- this returns a bool indicating success, ignore for now.
			-- plot the error rate on a graph
			_ <- plotLine "line.png" $ map (\(_, Vocab.TrainProgress itcount _, avg) -> (itcount, NN.calcAvg avg))
						statusupdates
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

