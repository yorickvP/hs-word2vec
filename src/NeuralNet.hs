module NeuralNet (
    NeuralNet
  , randomNetwork
  , featureVecArray
  , runWord
  , runNetwork
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
import Util

type DVec = Mt.Vector Double
type DMat = Mt.Matrix Double

data NeuralNet = NeuralNet !(IntMap DVec) !DMat
getFeat :: NeuralNet -> Int -> DVec
getFeat (NeuralNet x _) i = x IM.! i
updateNet :: NeuralNet -> Int -> DVec -> DMat -> NeuralNet
updateNet (NeuralNet x _) idx val out =
	NeuralNet (IM.insert idx val x) out


-- output is a V * feat matrix
runNetwork :: NeuralNet -> Int -> DVec
runNetwork net@(NeuralNet _ output) wordIdx = (getFeat net wordIdx) `Mt.vXm` output

-- exp(a_i) / sum(k)(exp(a_k))
-- use logsoftmax here?
softmax :: DVec -> Int -> Double
softmax a i = (expa Mt.@> i) / (Mt.sumElements expa)
	where expa = Mt.mapVector exp $ a
-- derivative over q_k: = (d(k,i) - s(a,k)) * s(a,i)
-- k and i can be swapped
-- from wikipedia/Softmax_function
softmax' :: DVec -> Int -> Int -> Double
softmax' a i k = ((delta k i) - softmax a k) * softmax a i
	where delta a b = if a == b then 1 else 0
logsoftmax :: DVec -> Int -> Double
logsoftmax a i = if (0 == softmax a i) then error "log of zero" else log $ softmax a i
-- derivative over q_k: s'(a, i, k) / s(a, i)
-- k and i can *not* be swapped
logsoftmax' :: DVec -> Int -> Int -> Double
logsoftmax' a i k = (softmax' a i k) / (softmax a i)


runWord :: NeuralNet -> (Int, Int) -> NeuralNet
runWord net@(NeuralNet _ output) (wordIdx, expected) = updateNet net wordIdx newfeat newout
	where
		feature = getFeat net wordIdx
		out = runNetwork net wordIdx
		softerr = 0 - (logsoftmax out expected)
		-- calculate the delta for the output layer
		-- the softmax can be seen as another layer on top of it with fixed weight 1
		-- this means this is also the modified error of the output layer
		-- delta_k = Err_k * g'(in_k)
		moderr = Mt.mapVector (* softerr) $ Mt.buildVector (Mt.dim out) (logsoftmax' out expected)
		-- calculate the weight adjustment between the output and projection
		-- W_(j,k) := W_(j,k) + alpha * a_j * delta_k
		newout = Mt.fromColumns $ imap processout $ Mt.toColumns output
		processout k feat = Mt.zipVectorWith (adjustprojoutcon (moderr Mt.@> k)) feature feat
		adjustprojoutcon d_k activation weight = weight + rate * d_k * activation
		-- calculate the propagation deltas for the projection layer
		delta_j = output `Mt.mXv` moderr
		-- calculate the new weights for the feature vector
		newfeat = check_size feature delta_j $ Mt.zipVectorWith (\w d -> w + rate * d) feature delta_j
		--output !! k ! j + rate * (feature ! j) * (moderr ! k)
		rate = 0.005 :: Double
		check_size a b x = if (Mt.dim a /= Mt.dim b) then error "size doesn't match" else x


randomNetwork :: Int -> Int -> IO NeuralNet
randomNetwork vocab dimen = do
	seeda <- randomIO :: IO Int
	seedb <- randomIO :: IO Int
	let a = Mt.randomVector seeda Mt.Uniform (vocab * dimen)
	let b = Mt.randomVector seedb Mt.Uniform (vocab * dimen)
	let ft = IM.fromAscList $ imap (,) (Mt.toRows $ Mt.reshape dimen a)
	return $ NeuralNet ft (Mt.reshape vocab b)


featureVecArray :: NeuralNet -> [DVec]
featureVecArray (NeuralNet ft_ _) = map snd $ IM.toAscList ft_

forceEval :: NeuralNet -> Int
forceEval (NeuralNet ft _) = IM.size $ ft
