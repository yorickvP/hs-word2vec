module PCA (
    pca,
    filterOutlier
    )
where

import qualified Data.Packed.Matrix as Mt
import qualified Data.Packed.Vector as Mt
import qualified Numeric.LinearAlgebra as Mt
import qualified Numeric.LinearAlgebra.Algorithms as Mt


type DVec = Mt.Vector Double
type DMat = Mt.Matrix Double

-- This PCA code is from an hmatrix example
-- Vector with the mean value of the columns of a matrix
mean a = Mt.constant (recip . fromIntegral . Mt.rows $ a) (Mt.rows a) Mt.<> a

-- covariance matrix of a list of observations stored as rows
cov x = (Mt.trans xc Mt.<> xc) / fromIntegral (Mt.rows x - 1)
    where xc = x - Mt.asRow (mean x)

-- creates the compression and decompression functions from the desired number of components
pca' :: Int -> DMat -> (DVec -> DVec , DVec -> DVec)
pca' n dataSet = (encode,decode)
  where
    encode x = vp Mt.<> (x - m)
    decode x = x Mt.<> vp + m
    m = mean dataSet
    c = cov dataSet
    (_,v) = Mt.eigSH' c
    vp = Mt.takeRows n (Mt.trans v)

pca :: Int -> [DVec] -> [DVec]
pca n dataSet = map enc dataSet
    where (enc, _dec) = pca' n (Mt.fromRows dataSet)

-- stddev for each dimension separately
stddev :: [DVec] -> DVec
stddev x = Mt.mapVector sqrt $ avg $ map squareall $ errors
    where
        squareall = Mt.mapVector (\a -> a * a)
        errors = map (\y -> y - (avg x)) x
        avg :: [DVec] -> DVec
        avg a = (sum a) / (fromIntegral $ length a)

-- filter points too far away from the mean
filterOutlier :: Double -> [DVec] -> (DVec -> Bool)
filterOutlier n x s = (sq (s - mean)) < (n*n* sq std)
    where
        std = stddev x
        mean = (sum x) / (fromIntegral $ length x)
        sq x = x `Mt.dot` x
