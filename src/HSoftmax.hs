module HSoftmax (
    NeuralNet
  , randomNetwork
  , runWord
  , getFeat
  , forceEval
  , Average
  , calcAvg
)
where

import System.Random
import qualified Numeric.LinearAlgebra as Mt
import Numeric.LinearAlgebra (dot, scale)
import Data.IntMap.Strict (IntMap, (!), insert, fromAscList, size)
import Data.List (foldl')
import Vocab (WordDesc (..), TrainProgress (..))
import Control.Monad.Writer
import Util

type DVec = Mt.Vector Double
type Features = DVec
type OutLayer = IntMap DVec
{-
Let's specify a type to store our entire network in.
Together with the Vocab, this'll be the state of the entire system.
-}
data NeuralNet = NeuralNet !(IntMap Features) !(OutLayer) !Average


-- furthermore, we're going to need functions for common operations
-- (exported) getFeat gets a feature vector for a specific word index
-- this is also used to output the word data
getFeat :: NeuralNet -> Int -> Features
getFeat (NeuralNet x _ _) i = x ! i
-- and this is a common case, too, a single new feature vector
-- and an updated set of output vectors
updateNet :: NeuralNet -> Int -> Features -> (OutLayer) -> Average -> NeuralNet
updateNet (NeuralNet x _ _) idx val out avg =
	NeuralNet (insert idx val x) out avg

-- generate a starting network, ft = rand(0,1) / dimen. ot = 0.
randomNetwork :: Int -> Int -> IO NeuralNet
randomNetwork vocab dimen = do
	seeda <- randomIO :: IO Int
	let a = Mt.randomVector seeda Mt.Uniform (vocab * dimen)
	let a' = a / (fromIntegral dimen)
	let b = Mt.constant 0.0 (vocab * dimen) :: DVec
	let ft = fromAscList $ imap (,) (Mt.toRows $ Mt.reshape dimen a')
	let ot = fromAscList $ imap (,) (Mt.toRows $ Mt.reshape dimen b)
	return $ NeuralNet ft ot cleanAvg


forceEval :: NeuralNet -> Int
forceEval (NeuralNet ft _ _) = size ft

-- A datatype for keeping track of averages (that are still being added to)
newtype Average = Average (Double, Int)

calcAvg :: Average -> Double
calcAvg (Average (total, len)) = total / (fromIntegral len)
addAvg :: Average -> Double -> Int -> Average
addAvg (Average (total, len)) subtot sublen = Average (total + subtot, len + sublen)
cleanAvg :: Average
cleanAvg = Average (0, 0)
avgSize :: Average -> Int
avgSize (Average (_, len)) = len

-- and some types for reporting the average error and progress
type StatusUpdate = (Double, TrainProgress, Average)
type StatusWriter = Writer [StatusUpdate]


-- the sigmoid function
sigmoid :: Double -> Double
sigmoid x = 1.0 / (1.0 + (exp (- x)))
{-
we are optimizing
sum(words) sum(context) log product(point,code)(sigmoid (code?1:-1) * (l1 . point))
using exp-sum-log instead of product:
sum(words) sum(context) log exp sum(point,code)(log sigmoid (code?1:-1) * (l1 . point))
the log/exp cancel out, so we can individually optimize log sigmoid(code?1:-1)*(l1.point)
using Stochastic Gradient Descent

huffman trees are useful because they minimize the calculations
(hierarchical softmax works with any kind of binary tree). huffman trees assign shorter
codes to more frequent words, so less calculations needed on a whole.


The derivatives of log sigmoid for both codes:
derivative of logsigmoid(1 * x) = 1 - sigmoid(x) (or sigmoid(-x))
derivative of logsigmoid(-1 * x) = 0 - sigmoid(x) (or -sigmoid(x))
-}
-- Let's define the rate for a particular part of training, to get a decreasing learning rate.
-- this might be more helpful with a bigger corpus, when there isn't much to be learned towards the end.
rateAdj :: (Double, Double) -> TrainProgress -> Double
rateAdj (rateMax, rateMin) (TrainProgress itcount total) =
	max rateMin $ rateMax * (1.0 - ((fromIntegral itcount) / (1.0 + fromIntegral total)))
{-
For every word a: for every some word b around it (this code is in Vocab), lookup a in a binary tree,
recalculate the learning rate, and perform gradient descent
on multiply the vector in layer 0(features) corresponding to word b, with each of the
vectors in layer 1 corresponding to the path to the top the binary tree (taking into consideration
if something is the left or right child).
While doing this, keep track of the average gradient descent error, and every 10000 iterations, write a
progress report.
The code below is mostly equivalent to the word2vec C code for skipgram hierarchical softmax
-}
runWord :: (Double, Double) -> NeuralNet -> TrainProgress -> (WordDesc, WordDesc) -> StatusWriter NeuralNet
runWord (rateMax, rateMin) net@(NeuralNet _ output avg) progress@(TrainProgress itcount _)
							  (WordDesc _ treepos, WordDesc expected _) = do
	-- write a progress report every 10000 iterations (if we didn't already, this function is
	-- used more than once per iteration)
	when ((avgSize avg' > 100) && ((itcount `mod` 10000) == 0)) $ tell [(rate, progress, avg')]
	return $ updateNet net expected newfeat newout newavg
	where
		rate    = rateAdj (rateMax, rateMin) progress
		l1      = getFeat net expected
		neu1e   = Mt.constant 0.0 (Mt.dim l1)
		(neu1e', newout, toterr) = foldl' (singleIter rate l1) (neu1e, output, 0.0) treepos
		newfeat = l1 + neu1e'
		-- keep track of the average error over some iterations
		avg'    = addAvg avg toterr (length treepos)
		newavg  = if (itcount `mod` 10000) == 0 then cleanAvg else avg'

-- this function runs for every node between the word and the top of the tree. We update the output
-- instantly, but sum the feature changes and update that when done (the output wouldn't be practical to
-- wait with, because it has to update for multiple points)
singleIter :: Double -> Features -> (DVec, OutLayer, Double) -> (Bool, Int) -> (DVec, OutLayer, Double)
singleIter rate l1 (neu1e, output, err) (c, p) = (neu1e', output', err + errf)
	where
		l2      = output ! p
		f       = l1 `dot` l2
		-- calculate the error
		errf    = log $ sigmoid $ (if c then -1.0 else (1.0)) * f
		--   g  = logsigmoid_c'(l1 `dot` l2) * rate
		g       = (1.0 - (fromIntegral $ fromEnum c) - (sigmoid f)) * rate
		neu1e'  = neu1e + (g `scale` l2)
		output' = insert p (l2 + (g `scale` l1)) output

