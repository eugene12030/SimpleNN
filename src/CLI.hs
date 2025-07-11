{-# LANGUAGE RecordWildCards #-}

module CLI
  ( Command(..)
  , parseCLI
  , runCLI
  ) where

import           Options.Applicative
import           Data.Char                  (toLower)
import qualified Data.List                  as List
import qualified Data.Vector                as V

import           Core                       ( Vector
                                            , Activation(..)
                                            , Network
                                            , initializeNetwork
                                            , predictUnit
                                            )
import           DataProcessing             ( loadCSV
                                            , splitDataset
                                            )

-- | Top-level commands
data Command
  = Init
      { cInputSize   :: Int
      , cLayerSizes  :: [Int]
      , cActivations :: [Activation]
      }
  | Train
      { cInputSize   :: Int
      , cLayerSizes  :: [Int]
      , cActivations :: [Activation]
      , cTrainCSV    :: FilePath
      , cValidRatio  :: Double
      , cEpochs      :: Int
      , cLearningRate:: Double
      }
  | Predict
      { cInputSize   :: Int
      , cLayerSizes  :: [Int]
      , cActivations :: [Activation]
      , cInputCSV    :: FilePath
      }
  | Eval
      { cInputSize   :: Int
      , cLayerSizes  :: [Int]
      , cActivations :: [Activation]
      , cTestCSV     :: FilePath
      }
  deriving (Show)

-- | CLI parser entrypoint
parseCLI :: ParserInfo Command
parseCLI = info (hsubparser
    ( command "init"    (initCmd    `withInfo` "Initialize a new network")
   <> command "train"   (trainCmd   `withInfo` "Train (stub) on CSV data")
   <> command "predict" (predictCmd `withInfo` "Predict labels for new inputs")
   <> command "eval"    (evalCmd    `withInfo` "Evaluate accuracy on test data")
    ))
  ( fullDesc
 <> progDesc "SimpleNN: neural-net CLI"
 <> header   "snn" )
  where
    withInfo p desc = info (p <**> helper) (progDesc desc)

-- | Common flags for init/train/predict/eval
initFlags :: Parser (Int, [Int], [Activation])
initFlags =
      (,,)
  <$> option auto
        ( long "input-size"
       <> short 'i'
       <> metavar "N"
       <> help "Number of input features" )
  <*> some (option auto
        ( long "layer"
       <> short 'l'
       <> metavar "SIZE"
       <> help "Hidden/output layer size (repeat per layer)" ))
  <*> some (option parseAct
        ( long "activation"
       <> short 'a'
       <> metavar "ReLU|Softmax|Identity"
       <> help "Activation for each layer (repeat for each --layer)" ))

-- | init command
initCmd :: Parser Command
initCmd = Init <$> option auto
                       ( long "input-size"
                      <> short 'i'
                      <> metavar "N"
                      <> help "Number of input features" )
               <*> some (option auto
                       ( long "layer"
                      <> short 'l'
                      <> metavar "SIZE"
                      <> help "Hidden/output layer size (repeat per layer)" ))
               <*> some (option parseAct
                       ( long "activation"
                      <> short 'a'
                      <> metavar "ReLU|Softmax|Identity"
                      <> help "Activation for each layer (repeat for each --layer)" ))

-- | train command
trainCmd :: Parser Command
trainCmd =
    Train
  <$> option auto
        ( long "input-size"
       <> short 'i'
       <> metavar "N"
       <> help "Number of input features" )
  <*> some (option auto
        ( long "layer"
       <> short 'l'
       <> metavar "SIZE"
       <> help "Hidden/output layer size (repeat per layer)" ))
  <*> some (option parseAct
        ( long "activation"
       <> short 'a'
       <> metavar "ReLU|Softmax|Identity"
       <> help "Activation for each layer (repeat for each --layer)" ))
  <*> strOption
        ( long "train"
       <> metavar "CSV"
       <> help "MNIST CSV (label + 784 pixels)" )
  <*> option auto
        ( long "valid-ratio"
       <> metavar "R"
       <> value 0.2
       <> showDefault
       <> help "Validation fraction (0–1)" )
  <*> option auto
        ( long "epochs"
       <> short 'e'
       <> metavar "N"
       <> help "Number of epochs (stub only)" )
  <*> option auto
        ( long "lr"
       <> metavar "RATE"
       <> help "Learning rate (stub only)" )

-- | predict command
predictCmd :: Parser Command
predictCmd =
    Predict
  <$> option auto
        ( long "input-size"
       <> short 'i'
       <> metavar "N"
       <> help "Number of input features" )
  <*> some (option auto
        ( long "layer"
       <> short 'l'
       <> metavar "SIZE"
       <> help "Hidden/output layer size (repeat per layer)" ))
  <*> some (option parseAct
        ( long "activation"
       <> short 'a'
       <> metavar "ReLU|Softmax|Identity"
       <> help "Activation for each layer (repeat for each --layer)" ))
  <*> strOption
        ( long "input"
       <> metavar "CSV"
       <> help "CSV of new inputs (label column ignored)" )

-- | eval command
evalCmd :: Parser Command
evalCmd =
    Eval
  <$> option auto
        ( long "input-size"
       <> short 'i'
       <> metavar "N"
       <> help "Number of input features" )
  <*> some (option auto
        ( long "layer"
       <> short 'l'
       <> metavar "SIZE"
       <> help "Hidden/output layer size (repeat per layer)" ))
  <*> some (option parseAct
        ( long "activation"
       <> short 'a'
       <> metavar "ReLU|Softmax|Identity"
       <> help "Activation for each layer (repeat for each --layer)" ))
  <*> strOption
        ( long "test"
       <> metavar "CSV"
       <> help "MNIST CSV for testing (label + pixels)" )

-- | Run the chosen command
runCLI :: Command -> IO ()
-- INIT just prints the empty network
runCLI Init{..} = do
  net <- initializeNetwork cInputSize cLayerSizes cActivations
  putStrLn "=== Initialized network ==="
  print net

-- TRAIN is currently a stub that just shows dataset sizes
runCLI Train{..} = do
  net0 <- initializeNetwork cInputSize cLayerSizes cActivations
  raw  <- loadCSV cTrainCSV
  let (trainV, validV) = splitDataset cValidRatio raw
  putStrLn $ "Loaded " ++ show (length raw) ++ " samples"
  putStrLn $ " → training:   " ++ show (length trainV)
  putStrLn $ " → validation: " ++ show (length validV)
  putStrLn "=== (Stub) Training complete ==="
  print net0  -- nothing really changed

-- PREDICT drops the first (label) column and runs forward pass
runCLI Predict{..} = do
  net   <- initializeNetwork cInputSize cLayerSizes cActivations
  raw   <- loadCSV cInputCSV
  let feats = map (V.toList . V.tail) raw
      preds = map (predictUnit net) feats
  putStrLn "=== Predictions ==="
  mapM_ print preds

-- EVAL compares predictions to true labels
runCLI Eval{..} = do
  net   <- initializeNetwork cInputSize cLayerSizes cActivations
  raw   <- loadCSV cTestCSV
  let pairs = [ (V.toList (V.tail v), round (V.head v))
              | v <- raw
              ]
      total = length pairs
      correct = length [ () | (x,y) <- pairs, predictUnit net x == y ]
      acc = if total == 0 then 0 else fromIntegral correct / fromIntegral total
  putStrLn $ "Test samples: " ++ show total
  putStrLn $ "Correct:      " ++ show correct
  putStrLn $ "Accuracy:     " ++ show (acc * 100) ++ "%"

-- | String → Activation
parseAct :: ReadM Activation
parseAct = eitherReader $ \s -> case List.map toLower s of
  "relu"     -> Right ReLU
  "softmax"  -> Right Softmax
  "identity" -> Right Identity
  _          -> Left $ "Unknown activation: " ++ s