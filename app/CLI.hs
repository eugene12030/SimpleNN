{-# LANGUAGE RecordWildCards #-}

module CLI (
    Command(..),
    parseCLI,
    runCLI
) where 

import            Options.Applicative
import qualified  Data.List   as List
import            Data.Char   (toLower)
import            Numeric.LinearAlgebra (Vector)

import SimpleNN.Core (
    Activation(..),
    Network,
    initializeNetwork,
    predictUnit
    )

-- Mock data: signatures only
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

-- Top‑level commands
data Command 
    = Init {
        cInputSize    :: Int,
        cLayerSizes   :: [Int],
        cActivations  :: [Activation]
    }
    | Train {
        -- same init flags here:
        cInputSize    :: Int,
        cLayerSizes   :: [Int],
        cActivations  :: [Activation],

        cEpochs       :: Int,
        cLearningRate :: Double,
        cTrainCSV     :: FilePath,
        cValidRatio   :: Double
    }
    | Predict {
        cInputSize    :: Int,
        cLayerSizes   :: [Int],
        cActivations  :: [Activation],

        cInputCSV     :: FilePath
    }
    | Eval {
        cInputSize    :: Int,
        cLayerSizes   :: [Int],
        cActivations  :: [Activation],
        cTestCSV      :: FilePath
    }
    deriving (Show)

-- CLI parser entrypoint
parseCLI :: ParserInfo Command
parseCLI = info (hsubparser
    ( command "init"    (initCmd    `withInfo` "Build and show a new network")
   <> command "train"   (trainCmd   `withInfo` "Init, load data & train")
   <> command "predict" (predictCmd `withInfo` "Init, load inputs & predict")
   <> command "eval"    (evalCmd    `withInfo` "Init, load test & evaluate")
    ))
  ( fullDesc
    <> progDesc "SimpleNN: CLI without file serialization"
    <> header   "snn" )
  where
    withInfo p desc = info (p <**> helper) (progDesc desc)

-- Common init‐flags parser
initFlags :: Parser (Int, [Int], [Activation])
initFlags =
    (,,)
    <$> option auto
        ( long "input-size"
       <> short 'i'
       <> metavar "N"
       <> help "Number of inputs" )
    <*> some (option auto
        ( long "layer"
       <> short 'l'
       <> metavar "SIZE"
       <> help "Hidden/output layer size (repeat per layer)" ))
    <*> some (option parseAct
        ( long "activation"
       <> short 'a'
       <> metavar "ReLU|Softmax|Identity"
       <> help "Activation for each layer (one per --layer)" ))

-- Parsing "init" command
initCmd :: Parser Command
initCmd = fmap (\(i,ls,as) -> Init i ls as) initFlags

-- Parsing "train" command
trainCmd :: Parser Command
trainCmd = 
    (\(i,ls,as) e lr fp vr -> Train i ls as e lr fp vr)
    <$> initFlags
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
       <> help "CSV of (features,label) pairs" )
    <*> option auto
        ( long "valid-ratio"
       <> metavar "R"
       <> value 0.2
       <> showDefault
       <> help "Validation split ratio (0–1)" )

-- Parsing "predict" command
predictCmd :: Parser Command
predictCmd =
    (\(i,ls,as) fp -> Predict i ls as fp)
    <$> initFlags
    <*> strOption
        ( long "input"
       <> metavar "CSV"
       <> help "CSV of feature-only rows" )

-- Parsing "eval" command
evalCmd :: Parser Command
evalCmd =
    (\(i,ls,as) fp -> Eval i ls as fp)
    <$> initFlags
    <*> strOption
        ( long "test"
       <> metavar "CSV"
       <> help "CSV of (features,label) pairs" )

-- Running each command
runCLI :: Command -> IO ()
runCLI Init{..} = do
    -- initialize network in IO
    net <- initializeNetwork cInputSize cLayerSizes cActivations
    putStrLn "=== Initialized network ==="
    print net

runCLI Train{..} = do
    -- initialize network
    net0 <- initializeNetwork cInputSize cLayerSizes cActivations

    -- load & split data
    samples <- loadData cTrainCSV
    let (trD, vaD) = splitDataset cValidRatio samples

    -- train
    trained <- trainNetwork net0 cEpochs cLearningRate trD vaD
    putStrLn "=== Training complete ==="
    print trained

runCLI Predict{..} = do
    -- initialize network
    net <- initializeNetwork cInputSize cLayerSizes cActivations

    -- load inputs
    inputs <- loadInputs cInputCSV
    let preds = map (predictUnit net) inputs

    putStrLn "=== Predictions ==="
    mapM_ print preds

runCLI Eval{..} = do
    -- initialize network
    net <- initializeNetwork cInputSize cLayerSizes cActivations

    -- load & evaluate
    samples <- loadData cTestCSV
    acc     <- evalNetwork net samples

    putStrLn $ "=== Accuracy: " ++ show (acc * 100) ++ "% ==="

-- Convert string to Activation
parseAct :: ReadM Activation
parseAct = eitherReader $ \s -> case List.map toLower s of
    "relu"     -> Right ReLU
    "softmax"  -> Right Softmax
    "identity" -> Right Identity
    _          -> Left $ "Unknown activation: " ++ s
