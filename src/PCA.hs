module PCA (
    pca
    )
where

import qualified Data.Packed.Matrix as Mt
import qualified Data.Packed.Vector as Mt
import qualified Numeric.LinearAlgebra as Mt
import qualified Numeric.LinearAlgebra.Algorithms as Mt


type DVec = Mt.Vector Double
type DMat = Mt.Matrix Double

--normalize_mean :: [Mt.Vector Double] -> [Mt.Vector Double]
--normalize_mean x = map (flip (-) mean) x where
--	mean = (sum x) / (fromIntegral $ length x)

--covariance_matrix :: [Mt.Vector Double] -> DMat
--covariance_matrix x = (sum $ map (\a -> (Mt.asColumn a) * (Mt.asRow a)) x) / (fromIntegral $ length x)

--eigenvectors :: Mt.Matrix Double -> [Mt.Vector Double]
--eigenvectors x = Mt.toColumns x where
--	(u, s, v) = Mt.svd x

--pca :: Int -> [DVec] -> [DVec]
--pca dims x = map (`Mt.vXm` base) dataset
--	where
--		base = Mt.fromColumns $ take dims $ eigenvectors $ covariance_matrix $ dataset
--		dataset = normalize_mean $ x

-- from an hmatrix example
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
    where (enc, dec) = pca' n (Mt.fromRows dataSet)
