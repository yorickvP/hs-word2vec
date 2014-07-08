
import Control.Monad.Random (evalRandT)
import Control.Monad.Writer
import System.Random (getStdGen)

import Data.List (intersperse)
import qualified Data.Packed.Vector as Mt

import Options.Applicative

import qualified Data.ByteString as B
import qualified Data.ByteString.Char8 as C8
import qualified Data.ByteString.Lazy as L
import qualified Vocab as Vocab
import qualified HSoftmax as NN
import Util

-- read the corpus.txt file, make it into a vocabulary
-- (but add a token for the end of the line to every line)
-- and then train over all of the words
main :: IO ()
main = do
	options <- execParser opts

	crps <- B.readFile $ filename options
	-- only use words that occur more than THRESH times
	let vocab = Vocab.makeVocab (Vocab.countWordFreqs $
		concatMap (++ [C8.pack "</s>"]) $ map C8.words $ C8.lines crps) (threshold options)
	putStrLn $ "Vocab loading complete: " ++
		(show $ Vocab.wordCount vocab) ++ " total words, "
		++ (show $ Vocab.uniqueWords vocab) ++ " unique words"
	-- DIMENS-dimensional vectors
	runAllWords options vocab crps (dimensions options)
	where
		opts = info (helper <*> trainargs)
		  ( fullDesc
		 <> progDesc "Train word vectors from a corpus")



runAllWords :: TrainArgs -> Vocab.Vocab -> B.ByteString -> Int -> IO ()
runAllWords options vocab content dimens = do
	-- starting with an empty net
	net  <- NN.randomNetwork (Vocab.uniqueWords vocab) dimens
	-- train with all the words
	net_ <- wordsIteration net
	putStrLn $ "training complete, writing to " ++ (outfile options)
	-- write all of the vectors sorted by word frequency to outwords.txt
	let sorted_vocab = Vocab.sortedVecList vocab (NN.getFeat net_)
	L.writeFile (outfile options) (
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
			-- get max lookaround and rates from options
			let func = Vocab.doIteration (subsamplerate options) vocab content (lookaround options)
												(NN.runWord (rateMax options, rateMin options)) net
			let (net2, statusupdates) = runWriter $ evalRandT func gen
			-- report on the progress from the writer monad (statusupdates is a lazy list)
			forM_ statusupdates (\(_rate, Vocab.TrainProgress itcount total, avg) ->
				putStrLn $ "iteration " ++ (show itcount) ++
								" / "   ++ (show total) ++
								" average: " ++ (show $ NN.calcAvg avg)
				)
			-- this returns a bool indicating success, ignore for now.
			-- plot the error rate on a graph
			_ <- plotLine "error.png" $ map (\(_, Vocab.TrainProgress itcount _, avg) -> (itcount, NN.calcAvg avg))
						statusupdates
			putStrLn $ "iteration " ++ (show itercount) ++ " complete "  ++ (show $ NN.forceEval net2)
			return net2
			-- possibly run this multiple times, not needed
			--if itercount < 10 then wordsIteration vocab net2 (itercount + 1) else return net2



data TrainArgs = TrainArgs
  { filename :: String
  , outfile  :: String
  , threshold :: Int
  , dimensions :: Int
  , rateMin  :: Double
  , rateMax  :: Double
  , lookaround :: Int
  , subsamplerate :: Maybe Float }

trainargs :: Parser TrainArgs
trainargs = TrainArgs
  <$> argument Just
      ( help "the file to read (corpus.txt)" <> metavar "FILENAME" <> value "corpus.txt" )
  <*> argument Just
      ( help "the output file (outwords.txt)" <> metavar "OUTFILE" <> value "outwords.txt" )
  <*> option
       ( long "threshold"
      <> help "the minimum amount of times a word occurs before it's used (5)"
      <> value 5
      <> metavar "THRESH" )
  <*> option
       ( long "dimensions"
      <> help "the amount of dimensions to train on (100)"
      <> value 100
      <> metavar "DIMENS" )
  <*> option
       ( long "rateMin"
      <> help "the minimum learning rate (0.001)"
      <> metavar "RATEMIN"
      <> value 0.001 )
  <*> option
       ( long "rateMax"
      <> help "the maximum learning rate (0.025)"
      <> metavar "RATEMAX"
      <> value 0.025 )
  <*> option
       ( long "lookaround"
      <> help "Set the max context size (5)"
      <> metavar "LOOKAROUND"
      <> value 5 )
  <*> optional (option
       ( long "subsample"
      <> metavar "SUBSAMPLERATE"
      <> help "The subsample rate (default: disabled. useful: 1e-5)" ))

