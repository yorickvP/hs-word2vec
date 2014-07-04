import PCA (pca)
import Util
import qualified Data.Packed.Vector as Mt
import Options.Applicative
import qualified Data.ByteString.Lazy.Char8 as BC8

main :: IO ()
main = do
	options <- execParser opts

	vecpairs <- readVectorsFile (filename options)
				(binary options) (limit options)
	let pcaoutput = pca 2 $ snd $ unzip vecpairs
	let e = zip (fst $ unzip vecpairs) pcaoutput
	plot $ map (\(x, vec) -> (BC8.unpack x, vec Mt.@> 0, vec Mt.@> 1)) e
	return ()
	where
		opts = info (helper <*> plotargs)
		  ( fullDesc
		 <> progDesc "Plot a word feature vocabulary file to an image")

data PlotArgs = PlotArgs
  { filename :: String
  , binary   :: Bool
  , limit    :: Maybe Int }

plotargs :: Parser PlotArgs
plotargs = PlotArgs
  <$> argument Just
      ( help "the file to read" <> metavar "FILENAME" )
  <*> switch
      ( long "binary"
     <> help "Whether it's binary" )
  <*> optional (option
      ( long "limit"
     <> metavar "LIMIT"
     <> help "Only plot the first entries" ))

