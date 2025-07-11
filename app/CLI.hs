{-# LANGUAGE RecordWildCards #-}

module CLI (
    Command(..),
    parseCLI,
    runCLI
) where 

import            Options.Applicative
import qualified  Data.List   as List
import            System.IO   (hPutStrLn, stderr)

import SimpleNN.Core (
    Activation(..),
    Network,
    initializeNetwork,
    forwardNetwork,
    predictUnit,
    )

-- Mock data:

-- in SimpleNN.Data
loadData    :: FilePath -> IO [(Vector Double, Int)]
loadInputs  :: FilePath -> IO [Vector Double]
splitDataset
  :: Double
  -> [(Vector Double, Int)]
  -> ([(Vector Double, Int)], [(Vector Double, Int)])
trainNetwork
  :: Network
  -> Int
  -> Double
  -> [(Vector Double, Int)]
  -> [(Vector Double, Int)]
  -> IO Network
evalNetwork
  :: Network
  -> [(Vector Double, Int)]
  -> IO Double

-- Mock data finish

--  Top‑level commands
data Command 
    = Init {
        cInputSize    ::  Int,
        cLayerSizes   :: [Int],
        cActivations  :: [Activation],
        cOutFile      :: FilePath
    }
    | Train {
        cModelIn      :: FilePath,
        cEpochs       :: Int,
        cLearningRate :: Double,
        cTrainCSV     :: FilePath,
        cValidRatio   :: Double,
        cModelOut     :: FilePath
    }
    | Train {
        cModelIn      :: FilePath,
        cEpochs       :: Int,
        cLearningRate :: Double,
        cTrainCSV     :: FilePath,
        cValidRatio   :: Double,
        cModelOut     :: FilePath
    }
    | Preduct {
        cModelIn      :: FilePath,
        cInputCSV     :: FilePath
    }
    | Eval {
        cModelIn      :: FilePath,
        cTestCSV      :: FilePath
    }
    deriving (Show)

-- CLI parser entrypoint

parseCLI :: ParserInfo Command
parseCLI = info (hsubparser
    ( command "init"    (initCmd    `withInfo` "Initialize a new network")
   <> command "train"   (trainCmd   `withInfo` "Train an existing network")
   <> command "predict" (predictCmd `withInfo` "Make predictions")
   <> command "eval"    (evalCmd    `withInfo` "Evaluate on test data")
    ))
  ( fullDesc
    <> progDesc "SimpleNN: a tiny Haskell neural‑net"
    <> header   "snn - neural network CLI" )
  where
    withInfo p desc = info (p <**> helper) (progDesc desc)

-- Parsing "init" command

initCmd :: Parser Command
initCmd = Init 
    <?> option auto
        ( long "input-size"
       <> short 'i'
       <> metavar "N"
       <> help "Number of inputs" )
    
    <*> some (option auto
        ( long "layer"
       <> short 'l'
       <> metavar "SIZE"
       <> help "Hidden/output layer size (specify once per layer)" ))
    
    <*> some (option parseAct
        ( long "activation"
       <> short 'a'
       <> metavar "ReLU|Softmax|Identity"
       <> help "Activation for each layer (one per --layer)" ))
    
    <*> strOption
        ( long "out"
       <> short 'o'
       <> metavar "FILE"
       <> value "network.txt"
       <> showDefault
       <> help "Where to write initial network" )

-- | Parsing "train" Command
trainCmd :: Parser Command
trainCmd = Train
    <$> strOption
        ( long "model-in"
       <> metavar "FILE"
       <> help "Serialized network file" )
    <*> option auto
        ( long "epochs"
       <> short 'e'
       <> metavar "N"
       <> help "Training epochs" )
    <*> option auto
        ( long "lr"
       <> metavar "RATE"
       <> help "Learning rate" )
    <*> strOption
        ( long "train"
       <> metavar "CSV"
       <> help "Training data CSV" )
    <*> option auto
        ( long "valid-ratio"
       <> metavar "R"
       <> value 0.2
       <> showDefault
       <> help "Validation split ratio (0–1)" )
    <*> strOption
        ( long "model-out"
       <> metavar "FILE"
       <> value "network-trained.txt"
       <> showDefault
       <> help "Where to write trained network" )

-- Parsing "predict" Command
predictCmd :: Parser Command
predictCmd = Predict
    <$> strOption
        ( long "model"
       <> metavar "FILE"
       <> help "Serialized network file" )
    <*> strOption
        ( long "input"
       <> metavar "CSV"
       <> help "CSV of new inputs" )

-- | Parsing "eval" Command
evalCmd :: Parser Command
evalCmd = Eval
  <$> strOption
        ( long "model"
       <> metavar "FILE"
       <> help "Serialized network file" )
  <*> strOption
        ( long "test"
       <> metavar "CSV"
       <> help "CSV of labeled test data" )

-- running each command

runCLI :: Command -> IO ()
runCLI Init{..} = do
  -- initialize network in IO
  net <- initializeNetwork cInputSize cLayerSizes cActivations
  writeFile cOutFile (show net)
  putStrLn $ "Network initialized → " ++ cOutFile


runCLI Train{..} = do
  -- load existing network
  txt <- readFile cModelIn
  let net = read txt :: Network
  -- load & split data
  datapoints <- loadCSV cTrainCSV
  let (trD, vaD) = splitDataset cValidRatio datapoints
  -- train
  trained <- trainNetwork net cEpochs cLearningRate trD vaD
  writeFile cModelOut (show trained)
  putStrLn $ "Training complete → " ++ cModelOut

runCLI Predict{..} = do
  txt <- readFile cModelIn
  let net = read txt :: Network
  datapoints <- loadCSV cInputCSV
  let inputs = map features datapoints
      preds  = map (predictUnit net) inputs
  putStrLn "Predictions:"
  mapM_ print preds

runCLI Eval{..} = do
  txt <- readFile cModelIn
  let net = read txt :: Network
  testD <- loadCSV cTestCSV
  acc   <- evalNetwork net testD
  putStrLn $ "Accuracy: " ++ show (acc * 100) ++ "%"

-- | Convert string to Activation
parseAct :: ReadM Activation
parseAct = eitherReader $ \s -> case List.map toLower s of
  "relu"     -> Right ReLU
  "softmax"  -> Right Softmax
  "identity" -> Right Identity
  _          -> Left $ "Unknown activation: " ++ s