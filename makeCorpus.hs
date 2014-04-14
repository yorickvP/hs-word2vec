import System.Process
import System.Exit
import Data.String.Utils
import Data.Char
import Data.Maybe (isJust)
import Data.List (find)
import System.Directory
import Control.Monad (forM_)

-- we need to run pdftotext for every pdf in pdfs, collect the output
-- and put it into a text file. then split that text file into sentences
-- filter out lines that have less than 10 characters without a . at the end
-- then discard sentences more than half starting with capitals
processFile :: String -> IO ()
processFile x = do
	putStrLn $ "processing " ++ x
	(code, out, err) <- readProcessWithExitCode "/usr/bin/pdftotext"
												[x, "/dev/stdout"] []
	case code of
		ExitSuccess -> do
			let (|>) = flip ($)
			let l = lines out |> map strip |> filter usefulline |> filter usefulsentence |>
					map removeref |> (map $ filter usefulchar) |> join " " |> split "." |>
					map strip |> filter usefulsentence |> unlines
			appendFile "corpus.txt" l
		_ -> return ()
	where
		usefulline [] = False
		usefulline x  = endswith "." x || length x > 10
		usefulchar x  = isLetter x || isNumber x || x == '.' || x == ' '
		stringhasletter x = isJust $ find (isLetter) x
		-- strip all of the [x] things, if they have no letters
		removeref []   = []
		removeref ('[':b) = case (span (/= ']') b) of
			(a, ']':r) -> if stringhasletter a
				then '[':a ++ "]" ++ removeref r
				else removeref r
			_ -> '[':b
		removeref  (a:b) = a : removeref b
		usefulsentence [] = False
		usefulsentence x = (length (filter (isUpper . (!! 0)) words)) * 2 < (length words)
			where words = filter (not . null) $ splitWs x
main :: IO ()
main = do
	fexist <- doesFileExist "corpus.txt"
	if fexist then removeFile "corpus.txt" else return ()
	pdfs <- getDirectoryContents "pdfs"
	forM_ (map ("pdfs/" ++) pdfs) processFile

