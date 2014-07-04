module Util (
	imap
  , plot
  , plotLine
  , readVectorsFile
)
where

import System.Process (rawSystem)
import System.Exit (ExitCode (ExitSuccess))
import Data.String.Utils
import qualified Data.ByteString.Lazy as BS
import qualified Data.ByteString.Lazy.Char8 as BC8
import Data.Binary.Get
import Data.Binary.IEEE754
import Control.Monad hiding (join)
import GHC.Float
import qualified Data.Packed.Vector as Mt


imap' :: (Int -> a -> b) -> Int -> [a] -> [b]
imap' _ _ []     = []
imap' f i (x:xs) = f i x : imap' f (i + 1) xs
imap = flip imap' 0


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

plotLine :: (Show a, Num a) => [(Int, a)] -> IO Bool
plotLine points = do
	writeFile filename dataset
	exitCode <- rawSystem "gnuplot" args
	return $ exitCode == ExitSuccess
	where 
		-- todo see if this works when haskell uses scientific notation
		dataset = unlines $ map (\(a, b) -> (show a) ++ " " ++ (show b)) points
		args = ["-e", join ";" [
				"set term png size 1024,1024",
				--"set offsets 1,1,1,1",
				"set output \"line.png\"",
				"plot \"" ++ filename ++ "\" using 1:2 with linespoints title \"\""]]
		filename = "plot_line.dat"
maybeTake :: Maybe Int -> [a] -> [a]
maybeTake Nothing a = a
maybeTake (Just x) a = take x a

readUp :: Char -> BS.ByteString -> (BS.ByteString, BS.ByteString)
readUp chr str = let (a, b) = BC8.break (== chr) str
				 in (a, BS.drop 1 b)

readVectorsFile :: String -> Bool -> Maybe Int -> IO [(BS.ByteString, Mt.Vector Double)]
readVectorsFile filename binary limit = do
	filecontent <- BS.readFile $ filename
	let (firstLine, vecs) = readUp '\n' filecontent
	let [_, Just (numVecs, _)] = fmap BC8.readInt $ BC8.words firstLine
	putStrLn $ "reading " ++ (show numVecs) ++ "-dimensional vectors"
	return $ if binary then
		let vecpairs = maybeTake limit $ iterate numVecs vecs
		    fieldlist = map (Mt.fromList . snd) vecpairs
		in  zip (map fst vecpairs) fieldlist -- add the words back on
	else
		let cleanlines = maybeTake limit $ BC8.lines vecs
		    fieldlist = map readFields cleanlines
		in
			zip (map (head . BC8.words) cleanlines) fieldlist
	where
		readFields x = Mt.fromList (map (read . BC8.unpack) $ tail $ BC8.words x :: [Double])
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
