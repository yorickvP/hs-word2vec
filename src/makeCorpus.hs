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
-- or filter them based on the previous and next lines, so we get paragraphs
-- then discard sentences more than half starting with capitals
processFile :: String -> IO ()
processFile x = do
	putStrLn $ "processing " ++ x
	(code, out, err) <- readProcessWithExitCode "/usr/bin/pdftotext"
												[x, "/dev/stdout"] []
	case code of
		ExitSuccess -> do
			let (|>) = flip ($)
			let l = lines out |> map strip |> filterN 1 usefulline |> filter usefulsentence |>
					map removeref |> (map $ filter usefulchar) |>
					join " " |> split "." |>
					map strip |> filter usefulsentence |> 
					map tokenWords |>
					unlines
			appendFile "corpus.txt" l
		_ -> return ()
	where
		usefulline _ [] _        = False
		usefulline (a:_) x (b:_) = (length a > 20) || (length x > 20)
		usefulline _ x _         = endswith "." x || length x > 10
		usefulchar x  = isLetter x || isNumber x || x == '.' || x == ' '
		stringhasletter x = isJust $ find (isLetter) x
		-- strip all of the "[x]"" things, if they have no letters
		removeref []   = []
		removeref ('[':b) = case (span (/= ']') b) of
			(a, ']':r) -> if stringhasletter a
				then '[':a ++ "]" ++ removeref r
				else removeref r
			_ -> '[':b
		removeref  (a:b) = a : removeref b
		usefulsentence [] = False
		usefulsentence x = (length (filter (isUpper . (!! 0)) words)) * 2 < (length words)
			&& length x > 15
			&& (sum $ map length words) `div` (length words) < length x `div` 2
			where words = filter (not . null) $ splitWs x
		tokenWords line = join " " $ map tokenWord $ words line
		tokenWord x
			| any isSymbol x      = "<symbol>"
			| any isNumber x      = "<number>"
			| not (all isAscii x) = "<nonascii>"
			| otherwise           = map toLower x
main :: IO ()
main = do
	fexist <- doesFileExist "corpus.txt"
	if fexist then removeFile "corpus.txt" else return ()
	pdfs <- getDirectoryContents "pdfs"
	forM_ (map ("pdfs/" ++) pdfs) processFile

lastN :: Int -> [a] -> [a]
lastN len arr = drop ((length arr) - len) arr
-- iterate over an n*2 window around the element and filter based on that
filterN :: Int -> ([a] -> a -> [a] -> Bool) -> [a] -> [a]
filterN c pred (a:al)   = filterN' c [] a al pred
filterN c pred []       = []
filterN' c before x [] pred
  | pred before x []    = [x]
  | otherwise           = []
filterN' c before x after@(n:an) pred
  | pred before x after = x : filterN' c ((lastN (c-1) before)++[x]) n an pred
  | otherwise           =     filterN' c ((lastN (c-1) before)++[x]) n an pred
