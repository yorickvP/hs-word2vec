{-# LANGUAGE Arrows #-}
import Network.HTTP.Conduit
import System.Environment (getArgs)
import System.Directory
import qualified Data.ByteString.Lazy as L
import Control.Monad.IO.Class (liftIO, MonadIO)
import Network.HTTP.Types.Status (statusCode, Status)
import Network.HTTP.Types.Header (ResponseHeaders)
import Network.HTTP.Types.URI (parseQuery, renderQuery)
import Control.Exception (throw)
import Control.Monad (liftM, join, forM_)
import Data.Maybe (catMaybes, listToMaybe, maybeToList, maybe)
import Data.List (takeWhile, isInfixOf)
import Data.Conduit (($$), ($$+-))
import Data.Conduit.Binary (sinkFile)
import qualified Data.CaseInsensitive as CI
import Data.ByteString (ByteString)
import Data.ByteString.Char8 (unpack, pack)
import Network.URI
import Text.XML.HXT.Core
import Data.List.Utils (replace)
import qualified Codec.Binary.UTF8.String as UTF8

-- make a nice API for the arxiv search box
-- example url: http://search.arxiv.org:8081/?query=%22big+data%22+OR+cloud+OR+%22machine+learning%22+OR+%22artificial+intelligence%22+OR+%22distributed+computing%22&startat=450
-- todo: use another pack/unpack
makeQueryUrl :: String -> Int -> String
makeQueryUrl query start = "http://search.arxiv.org:8081/" ++ unpack (renderQuery True
	(map (\ (s, v) -> (pack s, Just $ pack v)) [
		("query", query),
		("startat", show start)]))

-- the search results page has the following info:
-- - number of results, total results
-- - results (author, title (year), piece matching, url)
--    best bet is to extract ID from url, then make a PDF link out of it
data SearchResult = SearchResult { title :: String
								 , author:: String
								 , year  :: Int
                                 , match :: String
                                 , url   :: String }
                                 deriving (Show, Eq)
-- - prev/next links. these are usable to go to the next page, or we could look at the counter and increment startat
--    using these links might be more future proof, at least they are the intended way of navigation
--     (there's a search API, but not for full-text search)
--    also, the tinyurl starts at result 40, so the intention might be to make a more generic scraper that follows links and then looks at pdf links
--     implementing an arxiv search api feels like the nicest way to go
data SearchPage = SearchPage { results :: [SearchResult]
							 , prev    :: Maybe String
							 , next    :: Maybe String }
							 deriving Show
-- the actual scraping
-- I don't think I like HXT
getSearchResults :: MonadIO m => Request -> Manager -> m (Maybe SearchPage)
getSearchResults req manager = do
	res <- httpLbs req manager
	let body = responseBody res
	return $ listToMaybe $ runLA (hread >>> tosearchpage) $ (UTF8.decode $ L.unpack body)
	where
		-- get the prev/next links and the search results
		-- without the body filter at the beginning,
		-- it will return multiple results
		-- I'm sure this makes sense to someone
		tosearchpage = (deep (hasName "body")) >>>
			proc a -> do
				res     <- listA extractResults      -< a
				prev    <- listA (findLinkHref "Prev") -< a
				next    <- listA (findLinkHref "Next") -< a
				returnA -< SearchPage { results = res
									  , prev = liftM fixUrl (listToMaybe prev)
									  , next = liftM fixUrl (listToMaybe next) }
		-- get all the search results, they are nicely in <td class="snipp"> in various classnames
		extractResults = (deep (attrEq "class" "snipp") >>>
			proc a -> do
				title   <- textFromClassName "title"   -< a
				author  <- textFromClassName "author"  -< a
				year    <- textFromClassName "year"    -< a
				snippet <- textFromClassName "snippet" -< a
				url     <- textFromClassName "url"     -< a
				returnA -< SearchResult title author (read year) snippet url)
		attrEq name val = (getAttrValue name >>> isA (== val)) `guards` this
		-- concatenate all text nodes inside
		innerText       = listA (deep isText >>> getText) >>> arr concat
		-- byclass >> text is used extremely often in extractResults, so split it off
		textFromClassName cls = deep (attrEq "class" cls) >>> innerText
		textHas str      = (innerText >>> isA (str `isInfixOf`)) `guards` this
		findLinkHref str = deep (hasName "a" >>> textHas str >>> getAttrValue0 "href")
		-- this is a bit of a hack, but it works for now.
		-- it doesn't seem like hxt parses entities inside attributes, which it should
		   -- does it parse them at all?
		fixUrl = replace "&#38;" "&"

-- look at the search result url, then check out the prev and next pages
getAllSearchResults :: MonadIO m => Manager -> [String] -> [String] -> m ([SearchResult])
getAllSearchResults man alreadydone [] = return []
getAllSearchResults man alreadydone (url : toBeDone)
    -- this duplication check actually fails on the tinyurl, so that one's done twice
    -- this might be a problem. possible fix: use a Set to avoid duplicate results?
    -- other fix: iterate over the list. this is O(n^2) but not a problem now
    -- using a set for the results but not the alreadydone would be inconsistent
	| elem url alreadydone = getAllSearchResults man alreadydone toBeDone
	| otherwise            = case parseUrl url of
		Nothing  -> do
			liftIO $ print $ "URL failed to parse " ++ url
			nextIteration []
		Just req -> do
			liftIO $ putStrLn ("getting " ++ url)
			sresults <- getSearchResults req man
			case sresults of
				Nothing -> nextIteration []
				Just searchres -> do
					--liftIO $ print $ searchres
					-- at first I was recursing both ways,
					-- but that doesn't keep the alreadydone in order
					let p = maybeToList $ liftM resolveUrl $ prev searchres
					let n = maybeToList $ liftM resolveUrl $ next searchres
					res <- nextIteration (p ++ n)
					-- this makes sure every result is only included once
					-- still O(n^2), might want a set
					return $ (filter (flip notElem res) (results searchres)) ++ res
				where
					resolveUrl url = "http://search.arxiv.org:8081/" ++ url
		where
			alreadydone' = url : alreadydone
			nextIteration add = getAllSearchResults man alreadydone' (add ++ toBeDone)


-- we need to get the pdf url for all the http://arxiv.org/abs/xxxx.xxxx urls
-- which is http://arxiv.org/pdf/xxxx.xxxx
getPDFUrl :: String -> String
getPDFUrl = replace "abs" "pdf"

-- I'd rather cache the search results, because I need to run this a couple of times
getCacheAllSearchResultPDFs :: MonadIO m => Manager -> String -> m ([String])
getCacheAllSearchResultPDFs man start = do
	fexist <- liftIO $ doesFileExist file
	if fexist then do
		f <- liftIO $ readFile file
		return $ lines f
	else do
		searchres <- getAllSearchResults man [] [start]
		let pdfurls = map (getPDFUrl . url) searchres
		liftIO $ writeFile file (unlines pdfurls)
		return pdfurls
	where file = "search.txt"

-- next, download them all
--downloadPDF :: MonadIO m => Request -> Manager -> m ()
downloadPDF (url, req) manager = do
	fexist <- liftIO $ doesFileExist file
	if not fexist then do
		liftIO $ putStrLn $ "getting " ++ file
		res <- http (setUserAgent req) manager 
		liftIO $ createDirectoryIfMissing False directory
		responseBody res $$+- sinkFile $ file
	else return ()

	where
		filename :: String -> String
		filename = (++ ".pdf") . reverse . takeWhile (/= '/') . reverse
		directory = "pdfs/"
		file = directory ++ filename url

-- arxiv complains when downloading pdfs and there is no user agent
setUserAgent :: Request -> Request
setUserAgent req = req { requestHeaders = (CI.mk $ pack "User-Agent",
	pack "Haskell, downloading 400 or so pdfs"):requestHeaders req }

main :: IO ()
main = do
    args <- getArgs
    case args of
        [urlString] ->
			withManager $ \manager -> do
				pdfurls <- getCacheAllSearchResultPDFs manager urlString
				--liftIO $ print pdfurls
				forM_ (catMaybes $ map urlpair pdfurls) (flip downloadPDF manager)
             --   where
             --   	req' = req { redirectCount = 0 }
        _ -> putStrLn "Sorry, please provide exactly one URL"
	where
		-- make url into maybe (url, request)
		urlpair a = liftM ((,) a) $ parseUrl a

