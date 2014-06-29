import PCA (pca)
import Util
import qualified Data.Packed.Vector as Mt
import Options.Applicative
import Data.Maybe
import qualified Data.ByteString.Lazy as BS
import qualified Data.ByteString.Lazy.Char8 as BC8
import Data.Binary.Get
import Data.Binary.IEEE754
import Control.Monad
import GHC.Float

maybeTake :: Maybe Int -> [a] -> [a]
maybeTake Nothing a = a
maybeTake (Just x) a = take x a

readUp :: Char -> BS.ByteString -> (BS.ByteString, BS.ByteString)
readUp chr str = let (a, b) = BC8.break (== chr) str
				 in (a, BS.drop 1 b)

main :: IO ()
main = do
	opts <- execParser opts
	filecontent <- BS.readFile $ filename opts
	let (firstLine, vecs) = readUp '\n' filecontent
	let [_, Just (numVecs, _)] = fmap BC8.readInt $ BC8.words firstLine
	putStrLn $ "reading " ++ (show numVecs) ++ "-dimensional vectors"
	let e =
		if binary opts then
			let vecpairs = maybeTake (limit opts) $ iterate numVecs vecs
			    fieldlist = map (Mt.fromList . snd) vecpairs
			    pcaoutput = pca 2 fieldlist
			in
				zip (map (BC8.unpack . fst) vecpairs) pcaoutput -- add the words back on
		else
			let cleanlines = maybeTake (limit opts) $ map BC8.unpack $ BC8.lines vecs
			    fieldlist = map readFields cleanlines
			    pcaoutput = pca 2 fieldlist
			in
				zip (map (head . words) cleanlines) pcaoutput
				
	plot $ map (\(x, vec) -> (x, vec Mt.@> 0, vec Mt.@> 1)) e
	return ()
	where
		readFields x = Mt.fromList (map read $ tail $ words x :: [Double]) 
		opts = info (helper <*> plotargs)
		  ( fullDesc
		 <> progDesc "Plot a word feature vocabulary file to an image")
		iterate :: Int -> BS.ByteString -> [(BS.ByteString, [Double])]
		iterate noVecs start
			| BS.length (BS.take 2 start) < 2 = []
			| otherwise = let (rest, word, vecs) = parseBinary noVecs start
						  in  (word, map float2Double vecs) : iterate noVecs rest

parseBinary :: Int -> BS.ByteString -> (BS.ByteString, BS.ByteString, [Float])
parseBinary noVecs startStr = (rest', word, vectors)
	where
		-- remove leading \n
		vecs' = BC8.dropWhile (\a -> ((a == '\n') || (a == ' '))) startStr
		-- read until space
		(word, rest) = readUp ' ' vecs'
		-- read the things
		(vectors, rest') = runGet parseBin rest
		parseBin = liftM2 (,) (replicateM noVecs getFloat32le) getRemainingLazyByteString

			
data PlotArgs = PlotArgs
  { filename :: String
  , binary   :: Bool
  , limit    :: Maybe Int }

plotargs :: Parser PlotArgs
plotargs = PlotArgs
  <$> argument Just
      ( help "the file to read" <> metavar "FILENAME" )
  <*> switch
      ( long "binary"
     <> help "Whether it's binary" )
  <*> optional (option
      ( long "limit"
     <> metavar "LIMIT"
     <> help "Only plot the first entries" ))

