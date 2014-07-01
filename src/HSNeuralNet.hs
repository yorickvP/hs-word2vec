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
import qualified Numeric.LinearAlgebra.Algorithms as Mt
import Data.IntMap.Strict (IntMap)
import qualified Data.IntMap.Strict as IM
import Data.List (foldl')
import Util

type DVec = Mt.Vector Double
type DMat = Mt.Matrix Double

data NeuralNet = NeuralNet !(IntMap DVec) !(IntMap DVec)
getFeat :: NeuralNet -> Int -> DVec
getFeat (NeuralNet x _) i = x IM.! i
updateNet :: NeuralNet -> Int -> DVec -> (IntMap DVec) -> NeuralNet
updateNet (NeuralNet x _) idx val out =
	NeuralNet (IM.insert idx val x) out

sigmoid :: Double -> Double
sigmoid x = 1.0 / (1.0 + (exp (- x)))
-- output is a V * feat matrix
--runNetwork :: NeuralNet -> Int -> DVec
--runNetwork net@(NeuralNet _ output) wordIdx = (getFeat net wordIdx) `Mt.vXm` output



    --l1 = model.syn0[word2.index]
    --neu1e = [0.0] * model.layer1_size
    --for p in word.point:
    --    l2 = model.syn1[p]
    --    f = sigma(inner(l1 * l2))
    --    --f = sum(l1[x] * l2[x] for x in xrange(model.layer1_size))
    --    --f = 1.0 / (1.0 + exp(-f))
    --    g = (1 - word.code[d] - f) * alpha
    --    neu1e += g * l2
    --    l2 += g * l1
    --l1 += neu1e

-- we are optimizing
-- sum(words) sum(context) log exp sum(parents)(log sigmoid (code?1:-1) * (l1 . point))
-- the log/exp cancel out, so we can individually optimize log sigmoid(code?1:-1)*(l1.point)
-- for every code. huffman trees are useful because they minimize the calculations
-- (hierarchical softmax works with any kind of binary tree). huffman trees assign shorter
-- codes to more frequent words, so less calculations needed on a whole.
-- so calculate logsigmoid for every code and point
-- derivative of logsigmoid(1 * x) = 1 - sigmoid(x) (or sigmoid(-x))
-- derivative of logsigmoid(-1 * x) = 0 - sigmoid(x) (or -sigmoid(x))

-- foldl [(Bool, Int)] -> NeuralNet -> (neu1e, NeuralNet)
singleIter :: Double -> DVec -> (DVec, IntMap DVec) -> (Bool, Int) -> (DVec, IntMap DVec)
singleIter rate l1 (neu1e, output) (c, p) = (neu1e', output')
	where
		l2 = output IM.! p
		f  = sigmoid (Mt.dot l1 l2)
		-- g = logsigmoid'(l1 . l2) * rate
		g  = (1.0 - (bool2int c) - f) * rate
		neu1e' = Mt.add (Mt.scale g l2) neu1e
		output' = IM.insert p ((output IM.! p) + (Mt.scale g l1)) output
		bool2int True = 1.0
		bool2int False = 0.0

runWord :: NeuralNet -> Double -> (Int, Int, ([Bool], [Int])) -> NeuralNet
runWord net@(NeuralNet _ output) rate (wordIdx, expected, (upperC, upperI)) =
	updateNet net expected newfeat newout
	where
		l1 = getFeat net expected
		neu1e = Mt.constant 0.0 (Mt.dim l1)
		(neu1e', newout) = foldl' (singleIter rate l1) (neu1e, output) (zip upperC upperI)
		newfeat = Mt.add l1 neu1e'

		check_size a b x = if (Mt.dim a /= Mt.dim b) then error "size doesn't match" else x


randomNetwork :: Int -> Int -> IO NeuralNet
randomNetwork vocab dimen = do
	seeda <- randomIO :: IO Int
	seedb <- randomIO :: IO Int
	let a = Mt.randomVector seeda Mt.Uniform (vocab * dimen)
	let b = Mt.randomVector seedb Mt.Uniform (vocab * dimen)
	let ft = IM.fromAscList $ imap (,) (Mt.toRows $ Mt.reshape dimen a)
	let ot = IM.fromAscList $ imap (,) (Mt.toRows $ Mt.reshape dimen a)
	return $ NeuralNet ft ot


featureVecArray :: NeuralNet -> [DVec]
featureVecArray (NeuralNet ft_ _) = map snd $ IM.toAscList ft_

forceEval :: NeuralNet -> Int
forceEval (NeuralNet ft _) = IM.size $ ft
