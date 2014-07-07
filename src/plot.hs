import PCA (pca, filterOutlier)
import Util
import qualified Data.Packed.Vector as Mt
import Options.Applicative
import qualified Data.ByteString.Lazy.Char8 as BC8
import qualified Data.ByteString.Lazy as BS
import Data.Binary.Get
import Data.Binary.IEEE754
import Control.Monad
import GHC.Float

-- read a vector file
-- (also has support for the binary format of the original C program)
main :: IO ()
main = do
	options <- execParser opts

	vecpairs <- readVectorsFile (filename options)
				(binary options) (limit options)
	let pcainput = snd $ unzip vecpairs
	let pcaoutput = pca 2 pcainput
	let e = zip (fst $ unzip vecpairs) pcaoutput
	let f = case filtoutlier options of
		Nothing -> e
		Just thresh -> let outlierfilter = filterOutlier thresh pcaoutput in
						filter (outlierfilter . snd) e
	plot "pca.png" $ map (\(x, vec) -> (BC8.unpack x, vec Mt.@> 0, vec Mt.@> 1)) f
	return ()
	where
		opts = info (helper <*> plotargs)
		  ( fullDesc
		 <> progDesc "Plot a word feature vocabulary file to an image")

data PlotArgs = PlotArgs
  { filename :: String
  , binary   :: Bool
  , limit    :: Maybe Int
  , filtoutlier :: Maybe Double }

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
  <*> optional (option
      ( long "filter"
     <> metavar "THRESHOLD"
     <> help "Filter the points with more than THRESHOLD times the stdev length(squared)" ))

maybeTake :: Maybe Int -> [a] -> [a]
maybeTake Nothing a = a
maybeTake (Just x) a = take x a

-- head . BC8.split chr
readUp :: Char -> BS.ByteString -> (BS.ByteString, BS.ByteString)
readUp chr str = let (a, b) = BC8.break (== chr) str
				 in (a, BS.drop 1 b)

readVectorsFile :: String -> Bool -> Maybe Int -> IO [(BS.ByteString, Mt.Vector Double)]
readVectorsFile filename binary limit = do
	filecontent <- BS.readFile $ filename
	-- the first line has dimensions and number of words
	let (firstLine, vecs) = readUp '\n' filecontent
	let [_, Just (numVecs, _)] = fmap BC8.readInt $ BC8.words firstLine
	putStrLn $ "reading " ++ (show numVecs) ++ "-dimensional vectors"
	return $ if binary then
		let vecpairs = maybeTake limit $ iterateBin numVecs vecs
		    fieldlist = map (Mt.fromList . snd) vecpairs
		in  zip (map fst vecpairs) fieldlist -- add the words back on
	else
		let cleanlines = maybeTake limit $ BC8.lines vecs
		    fieldlist = map readFields cleanlines
		in
			zip (map (head . BC8.words) cleanlines) fieldlist
	where
		readFields x = Mt.fromList (map (read . BC8.unpack) $ tail $ BC8.words x :: [Double])
		iterateBin :: Int -> BS.ByteString -> [(BS.ByteString, [Double])]
		iterateBin noVecs start
			| BS.length (BS.take 2 start) < 2 = []
			| otherwise = let (rest, word, vecs) = parseBinary noVecs start
						  in  (word, map float2Double vecs) : iterateBin noVecs rest

-- sometimes, the vectors are stored in binary representation
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

