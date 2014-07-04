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
  , TrainProgress (..)
  , Vocab
)
where

import System.Random
import Control.Monad (foldM) -- filterM, liftM
import Control.Monad.Trans
import Data.Maybe (isJust)
import Control.Monad.Supply
import Control.Monad.Random
import Data.Array.IArray

import Data.List (foldl')
import qualified Data.Traversable as T

import qualified Data.ByteString as B
import qualified Data.ByteString.Char8 as C8
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

To promote vector quality on words that occur less frequently (and having to train on
less words), words that occur frequently can be skipped over some of the time, with
another word from the context looked at instead, making sure we're still training
on the same amount of words that we would have without the skipping.

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
contains a label.
-}
type WordIdx      = Int
type WordCount    = Int
type WordCounts   = HashMap B.ByteString WordCount
type WordIdCounts = HashMap B.ByteString (WordIdx, WordCount)
{-
Then, we need a datastructure to map the indices back to words,
to be able to make the PCA plot.
-}
type IndexWordMap = IntMap B.ByteString
{-
Now that we have these structures, let's put them into a datatype for easy use.

Another thing that is useful to keep track of is the number of words, because it's O(n)
to find it from the datastructures
This also stores a huffman tree with the words in them, for the hierarchical softmax.
-}

data Vocab = Vocab { wordCounts   :: !WordIdCounts
                   , wordCount    :: !WordCount       -- total word count, sum of wordCounts
                   , vocabWords   :: !IndexWordMap 
               	   , vocabHuffman :: !VocabHuffman}

type TreeIdx = Int
type BinTreeLocation = [(Bool, TreeIdx)]
data WordDesc = WordDesc WordIdx BinTreeLocation

-- Let's make a type to store the progress (iteration / total)
-- To easily pass it around, and use it to calculate the learning rate
-- and write the average every some iterations
data TrainProgress = TrainProgress Int WordCount
-- First: a function to create a progress starting at 0
startProgress :: Vocab -> TrainProgress
startProgress = TrainProgress 0 . wordCount
-- increase progress (Enum has succ, instance?)
inc :: TrainProgress -> TrainProgress
inc (TrainProgress x total) = TrainProgress (x+1) total

-- an ugly function...
-- iterate over words in a list of lines, with context,
-- (n before and n after, with 1 <= n <= ctx)
-- applying the neccesary maps and filters along the way
iterateWordContexts :: (RandomGen g, Monad m)  =>
	[Array Int B.ByteString] -> Int -> TrainProgress -> (B.ByteString -> Maybe b) -> (B.ByteString -> Bool)
	 -> (TrainProgress -> b -> a -> B.ByteString -> m a) -> a -> RandT g m a
iterateWordContexts (arr:xs) ctx progress primary_filter filt folder start = do
	-- iterate over array
	(progress', result) <- foldM (idxfold) (progress, start) $ indices arr
	-- iterate over the rest of the lines
	iterateWordContexts xs ctx progress' primary_filter filt folder result
	where
		-- given an index, loop over all of the context with the folder
		-- keeping track of progress as well.
		-- This type needs scoped type variables to work
		-- idxfold :: (RandomGen g) => (Int, a) -> Int -> Rand g (Int, a)
		idxfold (progress, strt) idx = do
			-- pick window size between 1 <= n <= ctx
			lookaround <- getRandomR (1, ctx)
			-- but make sure not to get more words than there are in the sentence
			let (lower, upper) = bounds arr
			-- the primary filter looks up the word in the vocabulary
			case primary_filter (arr ! idx) of
				Nothing   -> return (progress,  strt) -- non-filtered words are
					--not counted in the total, so don't increment the progress
				Just word -> do
					let before = toWords lookaround (enumFromThenTo (idx - 1) (idx - 2) lower)
					let after  = toWords lookaround (enumFromThenTo (idx + 1) (idx + 2) upper)
					strt'  <- lift $ foldM (folder progress word) strt before
					strt'' <- lift $ foldM (folder progress word) strt' after
					return (inc progress, strt'')
			where
				toWords lookaround x = take lookaround $ filter filt $ map (arr !) x
-- at the end, return the fold result
iterateWordContexts [] _ _ _ _ _ start = return start

-- and now, some functions to make the vocabulary

countWordFreqs :: [B.ByteString] -> WordCounts
countWordFreqs = foldl' (flip (flip (HM.insertWith (+)) 1)) HM.empty

-- assign indices to the words
labelWords :: WordCounts -> WordIdCounts
labelWords arr = evalSupply (T.mapM (\x -> do { i <- supply; return (i, x) }) arr) [0..]

-- filter the word counts based on a minimum threshold (infrequent words are typos or garbage)
-- and make them into a vocabulary (label, sum, make index lookup, make huffman tree)
makeVocab :: WordCounts -> WordCount -> Vocab
makeVocab counts thresh = Vocab labelled (HM.foldl' (+) 0 newcounts) indices tree
	where
		!newcounts = HM.filter (>= thresh) counts
		!labelled = labelWords newcounts
		!tree     = assignCodes (buildTree labelled) [] IM.empty
		           -- IndexWordMap: mapping WordIdx -> ByteString
		!indices = HM.foldlWithKey' (\o k (i, _) -> IM.insert i k o) IM.empty labelled

-- probably O(n)
uniqueWords :: Vocab -> WordCount
uniqueWords = HM.size . wordCounts

-- make the corpus file into a list of sentences (array of words), appending </s> to
-- every sentence to train the end of the line too. this improves training quite a bit.
lineArr :: B.ByteString -> [Array Int B.ByteString]
lineArr = map (toArray . (++ [C8.pack "</s>"]) . C8.words) . C8.lines
	where toArray x = listArray (0, length x - 1) x

-- used primarily for subsampling, lineArr, but with a filterM on the words
--lineFiltMArr :: Monad m => (B.ByteString -> m Bool) -> B.ByteString -> m [Array Int B.ByteString]
--lineFiltMArr pred = mapM (liftM toArray . filterM pred . C8.words) . C8.lines
--	where toArray x = listArray (0, length x - 1) x

-- some vocab lookup functions

hasWord :: Vocab -> B.ByteString -> Bool
hasWord v = isJust . safeWordIdx v

wordIdx :: Vocab -> B.ByteString -> WordIdx
wordIdx v = fst . ((wordCounts v) HM.!)

safeWordIdx :: Vocab -> B.ByteString -> Maybe WordIdx
safeWordIdx v = fmap fst . (flip HM.lookup) (wordCounts v)

-- exported
findIndex :: Vocab -> WordIdx -> B.ByteString
findIndex vocab str = (vocabWords vocab) IM.! str

-- unused without subsampling:
--wordFreq :: Vocab -> B.ByteString -> Float
--wordFreq (Vocab tr total _ _) x = ((fromIntegral count) / (fromIntegral total))
--	where (_, count) = tr HM.! x

 --subsampling isn't desirable, because we're mostly looking at the most frequent words
 --instead of all the words
--subsample :: (RandomGen g, Monad m) => Vocab -> B.ByteString -> RandT g m Bool
--subsample vocab@(Vocab tr total _ _) x = if not $ hasWord vocab x then return False else do
--	r <- getRandom
--	let freq = wordFreq vocab x
--	let prob = max 0 $ 1.0 - sqrt (1e-5 / freq)
--	return $ not $ prob >= (r :: Float)

getWordDesc :: Vocab -> WordIdx -> WordDesc
getWordDesc vocab x = WordDesc x $ (vocabHuffman vocab) IM.! x

-- taking a vocabulary and training file, a window size and a training function,
-- fold over all the training word pairs
doIteration :: (RandomGen g, Monad m) => Vocab -> B.ByteString -> WordCount ->
				(a -> TrainProgress -> (WordDesc, WordDesc) -> m a) -> a -> RandT g m a
doIteration vocab str ctx folder net = do
	-- disable subsampling: it's not good for PCA image quality
	-- trainlines <- lineFiltMArr (subsample vocab) str
	let trainlines = lineArr str
	iterateWordContexts trainlines ctx (startProgress vocab) (safeWordIdx vocab) filt train net
	where
		filt = hasWord vocab
		train progress a net b = folder net progress
								  (getWordDesc vocab a, getWordDesc vocab $ wordIdx vocab b)

-- The Huffman Tree

-- it's a simple binary tree
data HuffmanTree a = Leaf WordCount a
                   | Branch WordCount (HuffmanTree a) (HuffmanTree a) a
                   deriving Show -- other instances might be nice, but not needed
-- we assign indices to the tree nodes
type IndexedTree = HuffmanTree TreeIdx
-- the tree stores word counts, a branch's count is the sum of the children's count
probMass :: HuffmanTree a -> WordCount
probMass (Leaf x _) = x
probMass (Branch x _ _ _) = x

-- take the two nodes with the smallest word counts, and add them together
-- then add the created branch back to the queue, and repeat until there's only one
-- node left.
buildTree :: WordIdCounts -> IndexedTree
buildTree counts = evalSupply (build queue) [0..]
	where
		-- make leaf nodes out of every word (the index of leafs is a WordIdx)
		queue = PQ.fromList $ map (\(k,v) -> (v,Leaf v k)) (HM.elems counts)
		-- then keep taking the smallest probability nodes and adding them together
		build :: PQ.PQueue WordCount IndexedTree -> Supply TreeIdx IndexedTree
		build x =
			case PQ.minView x of
				Just (v, x') ->
					case PQ.minView x' of
						-- until there's just one left, then return that
						Nothing -> return v
						Just (v', x'') -> do
							-- assign branch indices using the supply monad
							i <- supply
							build (PQ.add pm (Branch pm v v' i) x'')
							where pm = (probMass v) + (probMass v')
-- then, we want to recurse over the tree, storing the paths to the root
-- for every word (where the left child is code False, the right one is code True)
-- in an IntMap, the VocabHuffman, that we add to the vocabulary
type VocabHuffman = IM.IntMap BinTreeLocation
assignCodes :: IndexedTree -> BinTreeLocation -> VocabHuffman -> VocabHuffman
assignCodes node upper codes = case node of
	Leaf   _ key     -> IM.insert key upper codes
	Branch _ a b x ->
		(assignCodes     a ((False, x):upper)
			(assignCodes b ((True,  x):upper) codes))

-- when we're done training, we want to store the vectors in descending word frequency
-- so plot_word2vec can read them
-- todo: other sort, but we were using the priority queue anyways
sortedWIDList :: WordIdCounts -> (WordIdx -> a) -> [(B.ByteString, a)]
sortedWIDList counts mapper = reverse $ pqtolist queue
	where
		queue = PQ.fromList $ map (\(k, (i, c)) -> (c, (k, mapper i))) $ HM.toList counts
		pqtolist :: PQ.PQueue Int (B.ByteString, b) -> [(B.ByteString, b)]
		pqtolist q = case PQ.minView q of
			Just (v, x') -> v : pqtolist x'
			Nothing -> []
-- now define a function to get a sorted vector list directly from the vocab
-- exported
sortedVecList :: Vocab -> (WordIdx -> a) -> [(B.ByteString, a)]
sortedVecList = sortedWIDList . wordCounts
