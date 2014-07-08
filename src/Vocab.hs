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
import Control.Monad (foldM, filterM, liftM)
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
type WordString   = B.ByteString
type WordCount    = Int
type WordCounts   = HashMap WordString WordCount
type WordIdCounts = HashMap WordString (WordIdx, WordCount)
{-
Then, we need a datastructure to map the indices back to words,
to be able to make the PCA plot.
-}
type IndexWordMap = IntMap WordString
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
-- helper function for doIteration, given an array, for each element,
-- fold folder over at most ctx words in the context
-- (n before and n after, with 1 <= n <= ctx)
-- applying the neccesary maps and filters along the way, to ensure
-- that the word is in the vocabulary and to pass the word descriptor
iterateWordContexts :: (RandomGen g, Monad m, Enum n, Ix n)  =>
	Vocab -> Int -> (a -> TrainProgress -> (WordDesc, WordDesc) -> m a) ->
	(TrainProgress, a) -> Array n WordString -> RandT g m (TrainProgress, a)
iterateWordContexts vocab ctx folder startval arr =
	-- iterate over array
	foldM idxfold startval $ assocs arr
	where
		train progress a net b = folder net progress
								  (getWordDesc vocab a, getWordDesc vocab $ wordIdx vocab b)
		-- given an index, loop over all of the context with the folder
		-- keeping track of progress as well.
		-- idxfold :: (RandomGen g) => (TrainProgress, a) -> (n, WordString) -> Rand g (TrainProgress, a)
		idxfold inval@(progress, strt) (idx, wordStr) =
			-- the primary filter looks up the word in the vocabulary
			case safeWordIdx vocab wordStr of
				Nothing   -> return inval -- non-filtered words are not
					-- counted in the total, so don't increment the progress
				Just word -> do
					-- pick window size between 1 <= n <= ctx
					lookaround <- getRandomR (1, ctx)
					-- but make sure not to get more words than there are in the sentence
					let (lower, upper) = bounds arr
					-- all the word indices before and after
					let beforeafter = [[lower .. pred idx], [succ idx .. upper]]
					-- now filter them to get only `lookaround` valid words
					let fullcontext = map (take lookaround . filter (hasWord vocab) . map (arr !)) beforeafter
					strt'  <- lift $ foldM (foldM $ train progress word) strt fullcontext
					return (inc progress, strt')

-- taking a vocabulary and training file, a window size and a training function,
-- fold over all the training word pairs
-- exported
doIteration :: (RandomGen g, Monad m) => Maybe Float -> Vocab -> B.ByteString -> Int ->
				(a -> TrainProgress -> (WordDesc, WordDesc) -> m a) -> a -> RandT g m a
doIteration subsamplerate vocab str ctx folder net = do
	trainlines <- case subsamplerate of
					Just rate -> lineFiltMArr (subsample rate vocab) str
					Nothing   -> return $ lineArr str
	(_progress', result) <- foldM (iterateWordContexts vocab ctx folder) (startProgress vocab, net) trainlines
	return result

-- and now, some functions to make the vocabulary

countWordFreqs :: [WordString] -> WordCounts
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
lineArr :: B.ByteString -> [Array Int WordString]
lineArr = map (toArray . (++ [C8.pack "</s>"]) . C8.words) . C8.lines
	where toArray x = listArray (0, length x - 1) x

-- used for subsampling: lineArr, but with a filterM on the words
lineFiltMArr :: Monad m => (WordString -> m Bool) -> B.ByteString -> m [Array Int WordString]
lineFiltMArr pred = mapM (liftM toArray . filterM pred . C8.words) . C8.lines
	where toArray x = listArray (0, length x - 1) x

-- some vocab lookup functions

hasWord :: Vocab -> WordString -> Bool
hasWord v = isJust . safeWordIdx v

wordIdx :: Vocab -> WordString -> WordIdx
wordIdx v = fst . ((wordCounts v) HM.!)

safeWordIdx :: Vocab -> WordString -> Maybe WordIdx
safeWordIdx v = fmap fst . (flip HM.lookup) (wordCounts v)

-- exported
findIndex :: Vocab -> WordIdx -> WordString
findIndex vocab str = (vocabWords vocab) IM.! str

-- Used for subsampling:
wordFreq :: Vocab -> WordString -> Float
wordFreq (Vocab tr total _ _) x = ((fromIntegral count) / (fromIntegral total))
	where (_, count) = tr HM.! x

-- subsampling (disabled by default)
subsample :: (RandomGen g, Monad m) => Float -> Vocab -> WordString -> RandT g m Bool
subsample rate vocab@(Vocab tr total _ _) x = if not $ hasWord vocab x then return False else do
	r <- getRandom
	let freq = wordFreq vocab x
	let prob = max 0 $ 1.0 - sqrt (rate / freq)
	return $ not $ prob >= (r :: Float)

getWordDesc :: Vocab -> WordIdx -> WordDesc
getWordDesc vocab x = WordDesc x $ (vocabHuffman vocab) IM.! x

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
sortedWIDList :: WordIdCounts -> (WordIdx -> a) -> [(WordString, a)]
sortedWIDList counts mapper = reverse $ pqtolist queue
	where
		queue = PQ.fromList $ map (\(k, (i, c)) -> (c, (k, mapper i))) $ HM.toList counts
		pqtolist :: PQ.PQueue Int (WordString, b) -> [(WordString, b)]
		pqtolist q = case PQ.minView q of
			Just (v, x') -> v : pqtolist x'
			Nothing -> []
-- now define a function to get a sorted vector list directly from the vocab
-- exported
sortedVecList :: Vocab -> (WordIdx -> a) -> [(WordString, a)]
sortedVecList = sortedWIDList . wordCounts
