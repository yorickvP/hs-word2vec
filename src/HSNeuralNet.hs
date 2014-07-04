module HSNeuralNet (
    NeuralNet
  , randomNetwork
  , featureVecArray
  , runWord
  , getFeat
  , forceEval
)
where

import System.Random
import qualified Data.Packed.Matrix as Mt
import qualified Data.Packed.Vector as Mt
import qualified Numeric.LinearAlgebra as Mt
import Data.IntMap.Strict (IntMap)
import qualified Data.IntMap.Strict as IM
import Data.List (foldl')
import Vocab (WordDesc (..), TrainProgress (..))
import Util

type DVec = Mt.Vector Double
type Features = DVec
type OutLayer = IntMap DVec
{-
Let's specify a type to store our entire network in.
Together with the Vocab, this'll be the state of the entire system.
-}
data NeuralNet = NeuralNet !(IntMap Features) !(OutLayer)
-- furthermore, we're going to need functions for common operations
-- (exported) getFeat gets a feature vector for a specific word index
-- this is also used to output the word data
getFeat :: NeuralNet -> Int -> Features
getFeat (NeuralNet x _) i = x IM.! i
-- and this is a common case, too, a single new feature vector
-- and an updated set of output vectors
updateNet :: NeuralNet -> Int -> Features -> (OutLayer) -> NeuralNet
updateNet (NeuralNet x _) idx val out =
	NeuralNet (IM.insert idx val x) out

sigmoid :: Double -> Double
sigmoid x = 1.0 / (1.0 + (exp (- x)))

-- we are optimizing
-- sum(words) sum(context) log product(point,code)(sigmoid (code?1:-1) * (l1 . point))
-- using exp-sum-log instead of product:
-- sum(words) sum(context) log exp sum(point,code)(log sigmoid (code?1:-1) * (l1 . point))
-- the log/exp cancel out, so we can individually optimize log sigmoid(code?1:-1)*(l1.point)
-- huffman trees are useful because they minimize the calculations
-- (hierarchical softmax works with any kind of binary tree). huffman trees assign shorter
-- codes to more frequent words, so less calculations needed on a whole.
-- so calculate logsigmoid for every code and point
-- derivative of logsigmoid(1 * x) = 1 - sigmoid(x) (or sigmoid(-x))
-- derivative of logsigmoid(-1 * x) = 0 - sigmoid(x) (or -sigmoid(x))
(rateMax, rateMin) = (0.025, 0.0001) :: (Double, Double)
rateAdj :: TrainProgress -> Double
rateAdj (TrainProgress itcount total) =
	max rateMin $ rateMax * (1.0 - ((fromIntegral itcount) / (1.0 + fromIntegral total)))

-- foldl [(Bool, Int)] -> NeuralNet -> (neu1e, NeuralNet)
singleIter :: Double -> Features -> (DVec, OutLayer) -> (Bool, Int) -> (DVec, OutLayer)
singleIter rate l1 (neu1e, output) (c, p) =
	if (abs f >= 6) then (neu1e, output) else (neu1e', output')
	where
		l2      = output IM.! p
		f       = l1 `Mt.dot` l2
		f'       = sigmoid (l1 `Mt.dot` l2)
		--   g  = logsigmoid'(l1 `dot` l2) * rate
		g       = (1.0 - (fromIntegral $ fromEnum c) - f') * rate
		neu1e'  = neu1e + (g `Mt.scale` l2)
		output' = IM.insert p ((output IM.! p) + (g `Mt.scale` l1)) output

-- use a word pair (and label/place in the tree of the word we're looking around)
runWord :: NeuralNet -> TrainProgress -> (WordDesc, WordDesc) -> NeuralNet
runWord net@(NeuralNet _ output) progress (WordDesc _ treepos, WordDesc expected _) =
	updateNet net expected newfeat newout
	where
		rate             = rateAdj progress
		l1               = getFeat net expected
		neu1e            = Mt.constant 0.0 (Mt.dim l1)
		(neu1e', newout) = foldl' (singleIter rate l1) (neu1e, output) treepos
		newfeat          = Mt.add l1 neu1e'


randomNetwork :: Int -> Int -> IO NeuralNet
randomNetwork vocab dimen = do
	seeda <- randomIO :: IO Int
	let a = Mt.randomVector seeda Mt.Uniform (vocab * dimen)
	let a' = a / (fromIntegral dimen)
	let b = Mt.constant 0.0 (vocab * dimen) :: DVec
	let ft = IM.fromAscList $ imap (,) (Mt.toRows $ Mt.reshape dimen a')
	let ot = IM.fromAscList $ imap (,) (Mt.toRows $ Mt.reshape dimen b)
	return $ NeuralNet ft ot


featureVecArray :: NeuralNet -> [Features]
featureVecArray (NeuralNet ft_ _) = map snd $ IM.toAscList ft_

forceEval :: NeuralNet -> Int
forceEval (NeuralNet ft _) = IM.size $ ft
