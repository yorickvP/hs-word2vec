{-# LANGUAGE BangPatterns, TemplateHaskell #-}
--module Corpus (
--	Corpus
--  , empty
--  , findIndex
--  , addTokens
--  , wordPairs
--  , numWords
--)
--where
{- dependencies: random, containers, MonadRandom, random-shuffle,
               unordered-containers,  bytestring,    utf8-string,
                            deepseq,  deepseq-th                -}
import System.Random
import System.Random.Shuffle
import Control.Monad (liftM)
import Control.Monad.Random
import Control.DeepSeq
import Control.DeepSeq.TH
import Data.Foldable (concatMap, fold)
import Data.List (foldl', mapAccumL)
import qualified Data.Traversable as T

import qualified Data.ByteString as B
import qualified Data.ByteString.UTF8 as UTF8
import Data.IntMap.Strict (IntMap)
import qualified Data.IntMap.Strict as IM
import Data.HashMap.Strict (HashMap)
import qualified Data.HashMap.Strict as HM

{-
The corpus file is stored as a text file, where each line is a sentence.
In order to train the neural net, we need to be able to iterate over every word
in random order, and train the neural net for every word in the context
(where we sample the words closer to the source word more often).

The strategy applied here is that we take the input lines, and convert them
into a datastructure, mapping every word to an index,
where a list of pairs that need to be trained on can be easily generated from.
input: "the quick brown" , output: [(0, 1), (0, 2), (1, 0), (1,2), (2,0), (2,1)].
(in reality, the output would be shuffled and pairs like (0, 2) would be less likely)

After the neural net is done training, its output vectors are paired to the words
and written out to a file for gnuplot to plot.

First, we specify the datastructure that stores context and id for every word, as a HashMap.
Originally, I used a Trie here, but the implementation was bad, offsetting the theoretical
advantages.
-}                                      -- ( id, [([before], [after])])
type WordContextMap = HashMap B.ByteString (Int, [([Int],      [Int])])
{-
Then, we need a datastructure to map the indices back to words,
so that we can make nice graphs later on.
-}
type IndexWordMap = IntMap B.ByteString
{-
Now that we have these structures, let's put them into a datatype for easy use.
Let's also include the training context size (how many words to look around every word),
because it's really a property used to generate and use the corpus.
Another thing that is useful to keep track of is the number of words, 
because that's O(n) on the Trie -- (it actually is O(1) on the map, use that?)
TL;DR: the ints are C and size. C is word lookaround.
-}
data Corpus = Corpus !WordContextMap !IndexWordMap !Int !Int
	deriving Show
-- Derive NF data so we can call deepseq (but should we?)
$(deriveNFData ''Corpus)

{-
Let's start with the obvious operations and getters that we're gonna need.
-}
-- an empty corpus with a given training size
empty :: Int -> Corpus
empty c = Corpus HM.empty IM.empty c 0

-- a getter for the size of a corpus
numWords :: Corpus -> Int
numWords (Corpus _ _ _ size) = size

-- a way to get a word back from its index
findIndex :: Corpus -> Int -> B.ByteString
findIndex (Corpus _ amap _ _) i = amap IM.! i

{-
Now, we'll need to devise a way to add sentences to a word map, one at a time.
The first problem is that our sentences are strings, but our word-contexts are index-based.
We can't iterate from left to right, because the words on the right side don't have indices yet.
So, first give every word an index, then iterate over them again to store the context.
-}
-- addWord gives a word an index, and returns it
addWord :: Corpus -> B.ByteString -> (Corpus, Int)
addWord x@(Corpus orig origmap c size) key =
	-- first check if the world is already in the trie
	case HM.lookup key orig of
		-- if so, just return the index.
		Just (i, _) -> (x, i)
		-- otherwise, add the word, increment the size
		-- and make the old size the index.
		Nothing     -> (Corpus
							(HM.insert key (size, []) orig)
							(IM.insert size key origmap)
							c
						(size + 1), size)

-- add adds the context for a word to the corpus
-- note that you need to know the index for all of the words
-- but ideally, the word we're manipulating should be a ByteString
add :: Corpus -> Int -> [Int] -> [Int] -> Corpus
add (Corpus orig origmap c size) key prev next =
	Corpus newTree origmap c size
	where -- adding lots of ! here, did not work.
		!newTree = HM.insert strkey newVal orig
		!newVal = (i, (prev, next):ctxs)
		-- this lookup is ugly and can be prevented
		-- with a zip in addTokens. But it's O(log n).
		!strkey = origmap IM.! key
		--V if this fails you didn't call addWord first
		!(Just (i, ctxs)) = HM.lookup strkey orig

{-
The actual function adding the tokens for an entire sentence is split in two.
The first part adds all of the words so they have indices.
The second part iterates over the tokens, taking the context (C words to either side) and adding it.
-}

-- what should really happen here is zipping the indices back with the words
-- so add doesn't have to look up the index in the map.
-- however, there are enough O(log n) operations here that it does not have a priority.
addTokens :: Corpus -> [B.ByteString] -> Corpus
addTokens orig toks = addTokens' orig' [] toks'
	where (orig', toks') = mapAccumL addWord orig toks

-- iterate over all of the 'b' tokens, moving them one-by-one to the 'a' tokens
-- while making sure that 'a' stays of C length.
-- essentially fold with context.
addTokens' :: Corpus -> [Int] -> [Int] -> Corpus
addTokens' orig a ([b])  = add orig b a []
addTokens' orig@(Corpus _ _ c _) a (b:bl) = addTokens' newC newA bl
	where -- adding lots of ! here. did slightly work.
		!newC = add orig b a (take c bl)
		!newA = ((lastN' (c-1) a)++[b])

-- take the last N elements of an array
-- used as a utility function in some places
lastN :: Int -> [a] -> [a]
lastN len arr = drop ((length arr) - len) arr
-- from stackoverflow
lastN' :: Int -> [a] -> [a]
lastN' n xs = foldl' (const .drop 1) xs (drop n xs)

{-
Finally, the function that converts the corpus datastructure into a shuffled list of training pairs.
The alternative to using shuffleM would be implementing Fisher-Yates using ST, but I hope this is efficient enough.
Traverse over the trie and concatmap it.
This function currently runs out of memory, it seems.
Iterating over the trie seems hard (using a lot of stackspace?), maybe iterating over the map
would be better. Alternatively, I could iterate over the numbers,
look them up in the map and look that up in the trie, but that feels inefficient.
-}
--concatM' :: (Monad m) => (a -> m [b]) -> HashMap x a -> m [b]
concatM' f = fmap concat . mapM f . HM.elems
wordPairs :: (RandomGen g) => Corpus -> Rand g [(Int, Int)]
--wordPairs (Corpus trie _ c _) = 
--	concatMapM indivWordPairs trie >>= shuffleM
--	where
--		indivWordPairs (i, inst)        = concatMapM (instWordPairs i) inst
--		instWordPairs i (before, after) = do
--			-- choose a random number r, up to c. take r words before and r words after.
--			-- this makes sure that closer words are sampled more often.
--			-- TODO: subsampling of frequent words.
--			r <- getRandomR (1, c)
--			return $ map ((,) i) $ (lastN' r before) ++ take r after
wordPairs (Corpus trie _ c _) =
	concatM' indivWordPairs trie >>= shuffleM
	where
		indivWordPairs (i, inst)        = concatMapM (instWordPairs i) inst
		instWordPairs i (before, after) = do
			-- choose a random number r, up to c. take r words before and r words after.
			-- this makes sure that closer words are sampled more often.
			-- TODO: subsampling of frequent words.
			r <- getRandomR (1, c)
			return $ map ((,) i) $ (lastN' r before) ++ take r after
		concatMapM f = liftM fold . T.mapM f
-- a simple test function for iterating over a corpus
--test :: Corpus -> Int
--test (Corpus trie _ _ _) = sum' $ Tr.toListBy (\_ (_, a) -> length a) trie
--	where sum' = foldl' (+) 0

-- read in everything from a corpus.txt file.
-- like found on http://ospinio.yori.cc/corpus.txt
getFullCorpus :: IO Corpus
getFullCorpus = do
	f <- readFile "corpus.txt"
	let flines = map (map UTF8.fromString) $ map words $ lines f
	return $! foldl' (\crps ln -> crps `seq` addTokens crps ln) (empty 3) flines
test1 :: Corpus -> IO ()
test1 crps = do
	a <- evalRandIO $ wordPairs crps
	putStrLn $ "word pair count: " ++ (show $ length a)
test2 :: Corpus -> IO ()
test2 crps = do
	a <- evalRandIO $ wordPairs crps
	putStrLn $ "word pair count: " ++ (show $ length a)

main :: IO ()
main = do
	crps <- getFullCorpus
	--putStrLn $ "word pair count: " ++ (show $ test crps)
	putStrLn $ "corpus loading complete " ++ (show $ numWords crps)
	test1 crps
	test2 crps
	-- a <- evalRandIO $ wordPairs crps
