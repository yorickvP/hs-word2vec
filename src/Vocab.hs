{-# LANGUAGE BangPatterns #-}
module Vocab (
    countWordFreqs
  , lineArr
  , doIteration
  , makeVocab
  , wordCount
  , uniqueWords
  , findIndex
  , sortedVecList
  , WordDesc (..)
  , Vocab
)
where

import System.Random
import Control.Monad (liftM, foldM)
import Data.Maybe (isJust)
import Control.Monad.Supply
import Control.Monad.Random
import Data.Array.IArray

import Data.List (foldl')
import qualified Data.Traversable as T

import qualified Data.ByteString as B
import qualified Data.ByteString.Char8 as C8
import qualified Data.ByteString.UTF8 as UTF8
import Data.IntMap.Strict (IntMap)
import qualified Data.IntMap.Strict as IM
import Data.HashMap.Strict (HashMap)
import qualified Data.HashMap.Strict as HM
import qualified Data.PriorityQueue.FingerTree as PQ

{-
The corpus file is stored as a text file, where each line is a sentence.
In order to train the neural net, we need to be able to iterate over every word,
and train the neural net for every word in the context
(where we sample the words closer to the source word more often).

Words that occur frequently are skipped over some of the time, but the n words before
that word are looked at instead.

The strategy applied here is that we take the input lines, and convert them
into a datastructure, mapping every word to an index, and storing the frequency
of every word.

After the neural net is done training, its output vectors are paired to the words
and written out to a file for gnuplot to plot.

-}
{-
Let's specify a datastructure for the word counts and indices. We can't give them
indices immediately, because words that occur less than n times (with n typically
around 2-5) have to be filtered out first.
This is why we have a datastructure storing just the word counts, and one that also
countains a label.
-}
type WordIdx      = Int
type WordCount    = Int
type WordCounts   = HashMap B.ByteString WordCount
type WordIdCounts = HashMap B.ByteString (WordIdx, WordCount)
{-
Then, we need a datastructure to map the indices back to words,
so that we can make nice graphs later on.
-}
type IndexWordMap = IntMap B.ByteString
{-
Now that we have these structures, let's put them into a datatype for easy use.
Let's also include the training context size (how many words to look around every word),
because it's really a property used to generate and use the corpus.
Another thing that is useful to keep track of is the number of words, because it's O(n)
to find it from the datastructures
-}

data Vocab = Vocab { wordCounts   :: !WordIdCounts
                   , wordCount    :: !WordCount       -- total word count, sum of wordCounts
                   , vocabWords   :: !IndexWordMap 
               	   , vocabHuffman :: !VocabHuffman}

type TreeIdx = Int
type BinTreeLocation = [(Bool, TreeIdx)]
data WordDesc = WordDesc WordIdx BinTreeLocation

iterateWordContexts :: (RandomGen g) =>
	[Array Int B.ByteString] -> Int -> Int -> (B.ByteString -> Maybe b) -> (B.ByteString -> Rand g Bool)
	 -> (Int -> b -> a -> B.ByteString -> a) -> a -> Rand g a
iterateWordContexts (arr:xs) ctx iterationcount primary_filter filt folder start = do
	-- iterate over array
	--r <- getRandomR (1, ctx)
	(iterationcount', result) <- foldM (idxfold) (iterationcount, start) $ indices arr
	-- get result
	iterateWordContexts xs ctx iterationcount' primary_filter filt folder result
	where
		-- need scoped type variables
		-- idxfold :: (RandomGen g) => (Int, a) -> Int -> Rand g (Int, a)
		idxfold (iterationcount, strt) idx = do
			lookaround <- getRandomR (1, ctx)
			let (lower, upper) = bounds arr
			case primary_filter (arr ! idx) of
				Nothing   -> return (iterationcount,  strt) -- should this be + 1?
				Just word -> do
					before <- wrds lookaround (enumFromThenTo (idx - 1) (idx - 2) lower)
					after  <- wrds lookaround (enumFromThenTo (idx + 1) (idx + 2) upper)
					let strt' = foldl' (folder iterationcount word) strt before
					return    $ (iterationcount + 1, foldl' (folder iterationcount word) strt' after)
			where
				--wrds :: (RandomGen g) => Int -> [Int] -> Rand g [B.ByteString]
				wrds lookaround x = takeNfiltM x (\i -> do
						let s = arr ! i
						out <- filt s
						return (if out then Just s else Nothing)) lookaround
iterateWordContexts [] _ _ _ _ _ start = return start

-- helper function, something like (take n $ catMaybes $ map pred xs), but monadically
-- without evaluating any unneeded results, like mapM would.
takeNfiltM :: (Monad m) => [a] -> (a -> m (Maybe b)) -> Int -> m [b]
takeNfiltM [] _ _ = return []
takeNfiltM  _ _ 0 = return []
takeNfiltM (x:xs) pred n = do
	x' <- pred x
	case x' of
		Nothing  -> takeNfiltM xs pred n
		Just val -> liftM (val :) $ takeNfiltM xs pred (n - 1)


countWordFreqs :: [B.ByteString] -> WordCounts
countWordFreqs = foldl' (flip (flip (HM.insertWith (+)) 1)) HM.empty

labelWords :: WordCounts -> WordIdCounts
labelWords arr = evalSupply (T.mapM (\x -> do { i <- supply; return (i, x) }) arr) [0..]
-- the int is the amount of times a word needs to occur before it's included
makeVocab :: WordCounts -> WordCount -> Vocab
makeVocab counts thresh = Vocab labelled (HM.foldl' (+) 0 newcounts) indices tree
	where
		!newcounts = HM.filter (>= thresh) counts
		!labelled = labelWords newcounts
		!tree     = assignCodes (buildTree labelled) [] IM.empty
		!indices = HM.foldlWithKey' (\o k (i, _) -> IM.insert i k o) IM.empty labelled

uniqueWords :: Vocab -> WordCount
uniqueWords = HM.size . wordCounts

lineArr :: B.ByteString -> [Array Int B.ByteString]
lineArr = map (toArray . C8.words) . UTF8.lines
	where toArray x = listArray (0, length x - 1) x


hasWord :: Vocab -> B.ByteString -> Bool
hasWord v = isJust . (flip HM.lookup) (wordCounts v)

wordIdx :: Vocab -> B.ByteString -> WordIdx
wordIdx v = fst . ((wordCounts v) HM.!)

safeWordIdx :: Vocab -> B.ByteString -> Maybe WordIdx
safeWordIdx v = fmap fst . (flip HM.lookup) (wordCounts v)

-- exported
findIndex :: Vocab -> WordIdx -> B.ByteString
findIndex vocab str = (vocabWords vocab) IM.! str

wordFreq :: Vocab -> B.ByteString -> Float
wordFreq (Vocab tr total _ _) x = ((fromIntegral count) / (fromIntegral total))
	where (_, count) = tr HM.! x

subsample :: (RandomGen g) => Vocab -> B.ByteString -> Rand g Bool
subsample vocab x = if not $ hasWord vocab x then return False else do
	r <- getRandom
	let freq = wordFreq vocab x
	let prob = 1 - (sqrt (1e-5 / freq))
	return $ prob >= (r :: Float)

getWordDesc :: Vocab -> WordIdx -> WordDesc
getWordDesc vocab x = WordDesc x $ (vocabHuffman vocab) IM.! x

doIteration :: (RandomGen g) => Vocab -> B.ByteString -> Int -> (Double, Double) ->
				(a -> Double -> (WordDesc, WordDesc) -> a) -> a -> Rand g a
doIteration vocab str ctx (rateMax, rateMin) folder net =
	iterateWordContexts (lineArr str) ctx 0 (safeWordIdx vocab) filt train net
	where
		filt = return . hasWord vocab --subsample vocab
		rateAdj :: Int -> Double
		rateAdj itcount = max rateMin $ rateMax * (1.0 - ((fromIntegral itcount) / (1.0 + fromIntegral (wordCount vocab))))
		train itcount a net b = folder net (rateAdj itcount)
								  (getWordDesc vocab a, getWordDesc vocab $ wordIdx vocab b)


data HuffmanTree a = Leaf WordCount a
                   | Branch WordCount (HuffmanTree a) (HuffmanTree a) a
                   deriving Show
type IndexedTree = HuffmanTree TreeIdx
probMass :: HuffmanTree a -> WordCount
probMass (Leaf x _) = x
probMass (Branch x _ _ _) = x

buildTree :: WordIdCounts -> IndexedTree
buildTree counts = evalSupply (build queue) [0..]
	where
		queue = PQ.fromList $ map (\(k,v) -> (v,Leaf v k)) (HM.elems counts)
		build :: PQ.PQueue WordCount IndexedTree -> Supply TreeIdx IndexedTree
		build x =
			case PQ.minView x of
				Just (v, x') ->
					case PQ.minView x' of
						Nothing -> return v
						Just (v', x'') -> do
							i <- supply
							build (PQ.add pm (Branch pm v v' i) x'')
							where pm = (probMass v) + (probMass v')
type VocabHuffman = IM.IntMap BinTreeLocation
assignCodes :: IndexedTree -> BinTreeLocation -> VocabHuffman -> VocabHuffman
assignCodes node upper codes = case node of
	Leaf   _ key     -> IM.insert key upper codes
	Branch _ a b x ->
		(assignCodes     a ((False, x):upper)
			(assignCodes b ((True,  x):upper) codes))


-- todo: other sort
sortedWIDList :: WordIdCounts -> (WordIdx -> a) -> [(B.ByteString, a)]
sortedWIDList counts mapper = reverse $ pqtolist queue
	where
		queue = PQ.fromList $ map (\(k, (i, c)) -> (c, (k, mapper i))) $ HM.toList counts
		pqtolist :: PQ.PQueue Int (B.ByteString, b) -> [(B.ByteString, b)]
		pqtolist q = case PQ.minView q of
			Just (v, x') -> v : pqtolist x'
			Nothing -> []

sortedVecList :: Vocab -> (WordIdx -> a) -> [(B.ByteString, a)]
sortedVecList = sortedWIDList . wordCounts
