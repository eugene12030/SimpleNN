{-# LANGUAGE OverloadedStrings #-}

module DataProcessing where

import qualified Data.Vector                as V
import qualified Data.ByteString.Lazy       as BL
import qualified Data.ByteString.Lazy.Char8 as BL8
import qualified Data.ByteString.Char8      as BS
import qualified Data.Csv                   as Csv
import           System.Random.Shuffle      (shuffleM)

-- | Data type to represent a single prediction record for CSV output
data PredictionRecord = PredictionRecord
  { recordNumber   :: Int
  , predictedValue :: Int
  }

-- | How to encode a PredictionRecord into a CSV row
instance Csv.ToRecord PredictionRecord where
  toRecord (PredictionRecord num predVal) = Csv.record
    [ BS.pack (show num)
    , BS.pack (show predVal)
    ]

-- | How to encode a PredictionRecord as a CSV header
instance Csv.ToNamedRecord PredictionRecord where
  toNamedRecord (PredictionRecord num predVal) = Csv.namedRecord
    [ "Number"   Csv..= num
    , "Prediction" Csv..= predVal
    ]

-- | Specify the order of fields for CSV encoding
instance Csv.DefaultOrdered PredictionRecord where
  headerOrder _ = Csv.header ["Number", "Prediction"]

loadCSV :: FilePath -> IO [V.Vector Double]
loadCSV filepath = do
    csvData <- BL.readFile filepath
    case Csv.decode Csv.NoHeader csvData of
        Left err -> do
            putStrLn $ "Error parsing CSV: " ++ err
            return []
        Right records -> do
            let vectors = V.toList $ V.map parseRecord records
            return $ filter (not . V.null) vectors
  where
    parseRecord :: V.Vector BL.ByteString -> V.Vector Double
    parseRecord record
        | V.length record >= 785 =
            let label  = readDouble (record V.! 0)
                pixels = V.map readDouble (V.slice 1 784 record)
            in V.cons label pixels
        | otherwise = V.empty

    readDouble :: BL.ByteString -> Double
    readDouble bs = case reads (BL8.unpack bs) of
        [(x, "")] -> x
        _         -> 0.0

-- | Normalize pixel values from 0–255 range to 0.0–1.0 range
normalizePixels :: [V.Vector Double] -> [V.Vector Double]
normalizePixels = map normalizeVector
  where
    normalizeVector :: V.Vector Double -> V.Vector Double
    normalizeVector v
        | V.null v  = v
        | otherwise =
            let lab  = V.head v
                px   = V.tail v
                norm = V.map (\x -> x / 255.0) px
            in V.cons lab norm

-- | Split dataset into training and validation sets
splitDataset :: Double -> [V.Vector Double] -> ([V.Vector Double], [V.Vector Double])
splitDataset ratio dataset = do
    let shuffled = dataset -- No shuffling here; use splitDatasetRandom for that
    let trainSize = round (ratio * fromIntegral (length dataset))
    splitAt trainSize shuffled

-- | Split dataset with random shuffling
splitDatasetRandom :: Double -> [V.Vector Double] -> IO ([V.Vector Double], [V.Vector Double])
splitDatasetRandom ratio dataset = do
    shuffled <- shuffleM dataset
    let trainSize = round (ratio * fromIntegral (length shuffled))
    return $ splitAt trainSize shuffled

writePredictionsCSV :: FilePath -> [Int] -> IO ()
writePredictionsCSV filepath predictions = do
  let records = zipWith PredictionRecord [0..] predictions
      csvData = Csv.encodeDefaultOrderedByName records
  BL.writeFile filepath csvData
  putStrLn $ "Predictions written to " ++ filepath