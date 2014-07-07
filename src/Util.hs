module Util (
	imap
  , plot
  , plotLine
)
where

import System.Process (rawSystem)
import System.Exit (ExitCode (ExitSuccess))
import Data.String.Utils

-- exported
imap' :: (Int -> a -> b) -> Int -> [a] -> [b]
imap' _ _ []     = []
imap' f i (x:xs) = f i x : imap' f (i + 1) xs
imap = flip imap' 0

-- I was using EasyPlot, but it didn't have the labels, so this is a hardcoded version of it
-- loosely based on http://hackage.haskell.org/package/easyplot-1.0/docs/src/Graphics-EasyPlot.html#Plot
plot :: (Show a, Num a) => String -> [(String, a, a)] -> IO Bool
plot imgfile points = do
	writeFile filename dataset
	exitCode <- rawSystem "gnuplot" args
	return $ exitCode == ExitSuccess
	where 
		-- todo see if this works when haskell uses scientific notation
		dataset = unlines $ map (\(s, a, b) -> s ++ " " ++ (show a) ++ " " ++ (show b)) points
		args = ["-e", join ";" [
				"set term png size 1024,1024 font \"Sans,8\"",
				--"set offsets 1,1,1,1",
				"set output \"" ++ imgfile ++ "\"",
				"plot \"" ++ filename ++ "\" using 2:3:1 with labels title \"\""]]
		filename = "plot1.dat"

plotLine :: (Show a, Num a) => String -> [(Int, a)] -> IO Bool
plotLine imgfile points = do
	writeFile filename dataset
	exitCode <- rawSystem "gnuplot" args
	return $ exitCode == ExitSuccess
	where 
		-- todo see if this works when haskell uses scientific notation
		dataset = unlines $ map (\(a, b) -> (show a) ++ " " ++ (show b)) points
		args = ["-e", join ";" [
				"set term png size 1024,1024",
				--"set offsets 1,1,1,1",
				"set output \"" ++ imgfile ++ "\"",
				"plot \"" ++ filename ++ "\" using 1:2 with linespoints title \"\""]]
		filename = "plot_line.dat"
