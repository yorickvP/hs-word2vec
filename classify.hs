
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
-- XXX: derivations of softmax have a k parameter, what is this for?

import qualified Data.Vector.Unboxed as V
import System.Random
import System.Random.Shuffle
import Control.Monad (forM_, foldM, liftM)
import Control.Monad.Random
import Debug.Trace
import Data.List (foldl)
import Data.Foldable (concatMap, fold)
import Text.Printf
import Data.String.Utils
import qualified Data.Packed.Matrix as Mt
import qualified Data.Packed.Vector as Mt
import qualified Numeric.LinearAlgebra as Mt
import qualified Numeric.LinearAlgebra.Algorithms as Mt
import qualified Data.Traversable as T
import qualified Data.Map as M
import System.Cmd (rawSystem)
import System.Exit (ExitCode (ExitSuccess))
type Vec = V.Vector Double
type Mat = [Vec]
-- output is a V * feat matrix
runNetwork :: Vec -> Mat -> Vec
runNetwork feature output = V.fromList $ map (dotProduct feature) output

-- exp(a_i) / sum(k)(exp(a_k))
softmax :: Vec -> Int -> Double
softmax a i = (expa V.! i) / (V.sum expa)
	where expa = V.map exp $ a
-- derivative: (d(k,i) - s(a,k)) * s(a,i)
-- k and i can be swapped, so I don't have to figure out
-- which does what
softmax' :: Vec -> Int -> Int -> Double
softmax' a i k = ((delta k i) - softmax a k) * softmax a i
	where delta a b = if a == b then 1 else 0
dotProduct :: Vec -> Vec -> Double
dotProduct a b = V.foldl (\l (a,b) -> l + a * b) 0 $ V.zip a b
logsoftmax :: Vec -> Int -> Double
logsoftmax a i = if (0 == softmax a i) then error "log of zero" else log $ softmax a i
logsoftmax' :: Vec -> Int -> Int -> Double
logsoftmax' a i k = (softmax' a i k) / (softmax a i)

--propagate :: Double -> Vec -> Vec -> Double
--propagate gin weights errs = gin * dotProduct weights errs
imap' :: (Int -> a -> b) -> Int -> [a] -> [b]
imap' _ _ []     = []
imap' f i (x:xs) = f i x : imap' f (i + 1) xs
imap = flip imap' 0

-- TODO: ew
transpose :: Mat -> Mat
transpose x = map (\i -> V.fromList $ map (V.! i) x) [0..(ml - 1)]
	where
		xl = length x
		ml = V.length (x !! 0)

runWord :: Vec -> Mat -> Int -> (Vec, Mat)
runWord feature output expected = (newfeat, newout)
	where
		out = runNetwork feature output
		softerr = 0 - (logsoftmax out expected)
		-- calculate the delta-k for the output layer
		-- the softmax can be seen as another layer on top of it with fixed weight 1
		-- this means this is also the modified error of the output layer
		moderr = V.map (* softerr) $ V.generate (V.length out) (logsoftmax' out expected)
		-- calculate the weight adjustment between the output and projection
		newout = imap processout output
		processout k feat = V.zipWith (adjustprojoutcon (moderr V.! k)) feature feat
		adjustprojoutcon d_k activation weight = weight + rate * d_k * activation
		-- calculate the propagation deltas for the projection layer
		delta_j = V.zipWith (*) (V.fromList $ map V.sum $ transpose output) moderr
		-- calculate the new weights for the feature vector
		newfeat = V.zipWith (\w d -> w + rate * d) feature delta_j
		--output !! k ! j + rate * (feature ! j) * (moderr ! k)
		rate = 0.005 :: Double
		traceShowId a = trace (show a) a


randomNetwork :: Int -> Int -> IO (Mat, Mat)
randomNetwork vocab dimen = do
	feature <- mapM (\_ -> randVec dimen) [0..(vocab - 1)]
	output  <- mapM (\_ -> randVec dimen) [0..(vocab - 1)]
	return (feature, output)
	where randVec x = V.generateM x (\_ -> randomIO)

--main = do
--	(fullfeat, output) <- randomNetwork 5 5
--	let feat0 = fullfeat !! 0
--	iter 0 feat0 output
--	where
--		iter x ft ot = do
--			let net = runNetwork ft ot
--			putStrLn $ "softmax0:" ++ (show $ softmax (runNetwork ft ot) 0)
--			putStrLn $ "softmax1:" ++ (show $ softmax (runNetwork ft ot) 1)
--			putStrLn $ "softmax2:" ++ (show $ softmax (runNetwork ft ot) 2)
--			putStrLn $ "softmax3:" ++ (show $ softmax (runNetwork ft ot) 3)
--			putStrLn $ "softmax4:" ++ (show $ softmax (runNetwork ft ot) 4)
--			--putStrLn $ "out:     " ++ (show $ ot)
--			let (ft2, ot2) = runWord ft ot 2
--			let (ft3, ot3) = runWord ft2 ot2 0
--			let (ft4, ot4) = runWord ft3 ot3 1
--			let (ft5, ot5) = runWord ft4 ot4 2
--			let (ft6, ot6) = runWord ft5 ot5 1
--			let (ft7, ot7) = runWord ft6 ot6 0
--			if (x > 5000) then return () else	iter (x + 1) ft7 ot7

wordStuff = M.fromList [
		("the",   (0, [([], ["quick", "brown", "fox"]), (["fox", "jumps", "over"], ["lazy", "dog"])])),
		("quick", (1, [(["the"], ["brown", "fox", "jumps"])])),
		("brown", (2, [(["the", "quick"], ["fox", "jumps", "over"])])),
		("fox",   (3, [(["the", "quick", "brown"], ["jumps", "over", "the"])])),
		("jumps", (4, [(["quick", "brown", "fox"], ["over", "the", "lazy"])])),
		("over",  (5, [(["brown", "fox", "jumps"], ["the", "lazy", "dog"])])),
		("lazy",  (6, [(["jumps", "over", "the"], ["dog"])])),
		("dog",   (7, [(["over", "the", "lazy"], [])]))
	]


-- from stackoverflow
lastN :: Int -> [a] -> [a]
lastN n xs = foldl (const .drop 1) xs (drop n xs)

-- convert a word list like above into shuffled pairs of training data
wordPairs :: (RandomGen g) => M.Map String (Int, [([String], [String])]) -> Int -> Rand g [(Int, Int)]
wordPairs words c = 
	concatMapM indivWordPairs words >>= shuffleM
	where
		indivWordPairs (i, inst)        = concatMapM (instWordPairs i) inst
		instWordPairs i (before, after) = do
			r <- getRandomR (1, c)
			return $ map ((,) i . fst . (words M.!)) $ (lastN r before) ++ take r after
		concatMapM f = liftM fold . T.mapM f

runWords :: (RandomGen g) => M.Map String (Int, [([String], [String])]) -> Int -> M.Map Int Vec -> Mat -> Rand g (M.Map Int Vec, Mat)
runWords words c feat out = do
	pairs <- wordPairs words c
	foldM runWordPair (feat, out) pairs
	where
		runWordPair (f, o) (a, b) = do
			let (thisf, newo) = runWord (f M.! a) o b
			return (M.insert a thisf f, newo)

runAllWords :: IO ()
runAllWords = do
	(feat, out) <- randomNetwork 8 8
	let ft = M.fromAscList $ imap (,) feat
	(ft_, ot_) <- iter (ft, out) 0
	putStrLn $ "complete: "
	putStrLn $ "feature: " ++ (show $ M.toList ft_)
	putStrLn $ "output:  " ++ (show $ ot_)
	let v = runNetwork (ft_ M.! 0) ot_
	let outs = V.imap (\i _ -> softmax v i) v
	putStrLn $ "full softmax output: " ++ (show $ map (printf "%.2f" :: Double->String) $ V.toList outs)
	putStrLn $ "PCA: " ++ (show $ length $ pca 2 ft_)
	let wordmap = M.fromList $ map (\(w, (x, _)) -> (x, w)) (M.toList wordStuff)
	a <- plot $ imap (\i x -> (wordmap M.! i, x Mt.@> 0, x Mt.@> 1) ) $ pca 2 ft_
	return ()
	where iter (ft, ot) x = do
		(ft2, ot2) <- evalRandIO $ runWords wordStuff 3 ft ot
		putStrLn $ "iteration " ++ (show x) ++ " complete " ++ (show $ sum $ map V.sum $ map snd $ M.toList $ ft2)
		let v = runNetwork (ft2 M.! 0) ot2
		let outs = V.imap (\i _ -> softmax v i) v
		putStrLn $ "full softmax output: " ++ (show $ map (printf "%.2f" :: Double->String) $ V.toList outs)
		putStrLn $ "network output     : " ++ (show $ map (printf "%.2f" :: Double->String) $ V.toList v)
		if x < 1000 then iter (ft2, ot2) (x + 1) else return (ft2, ot2)
-- plot "plot1.dat" using 2:3:1 with labels title ""
main = runAllWords
-- full softmax output: ["0.00","0.32","0.01","0.00","0.02","0.32","0.32","0.01"]
-- full softmax output: ["0.00","0.32","0.02","0.01","0.01","0.32","0.31","0.01"]
-- full softmax output: ["0.00","0.33","0.00","0.01","0.01","0.32","0.33","0.00"]

normalize_mean :: [Mt.Vector Double] -> [Mt.Vector Double]
normalize_mean x = map (flip (-) mean) x where
	mean = (sum x) / (fromIntegral $ length x)

covariance_matrix :: [Mt.Vector Double] -> Mt.Matrix Double
covariance_matrix x = (sum $ map (\a -> (Mt.asColumn a) * (Mt.asRow a)) x) / (fromIntegral $ length x)

eigenvectors :: Mt.Matrix Double -> [Mt.Vector Double]
eigenvectors x = Mt.toColumns x where
	(u, s, v) = Mt.svd x

pca :: Int -> M.Map Int Vec -> [Mt.Vector Double]
pca dims x = map (`Mt.vXm` base) dataset
	where
		base = Mt.fromColumns $ take dims $ eigenvectors $ covariance_matrix $ dataset
		dataset = normalize_mean $ toVecArray x
		toVecArray :: M.Map Int Vec -> [Mt.Vector Double]
		toVecArray x = map (Mt.fromList . V.toList . snd) $ M.toAscList x

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
				"set term png",
				"set offsets 1,1,1,1",
				"set output \"pca.png\"",
				"plot \"" ++ filename ++ "\" using 2:3:1 with labels title \"\""]]
		filename = "plot1.dat"
