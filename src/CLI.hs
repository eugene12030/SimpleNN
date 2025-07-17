{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleContexts #-}

module CLI
  ( Command(..)
  , parseCLI
  , runCLI
  ) where

import           Options.Applicative
import           Data.Char                   (toLower)
import qualified Data.List                   as List
import qualified Data.Vector                 as V
import           System.IO                   (hFlush, stdout)
import           System.Exit                 (exitSuccess)
import           Data.IORef                  (IORef, newIORef, readIORef, writeIORef) -- For mutable state

import           Core                        ( Vector
                                             , Activation(..)
                                             , Network
                                             , initializeNetwork
                                             , predictUnit
                                             )
import           DataProcessing              ( loadCSV
                                             , splitDataset
                                             )

-- | Top-level commands
data Command
  = Shell {}
  | Init
      { cInputSize   :: Int
      , cLayerSizes  :: [Int]
      , cActivations :: [Activation]
      }
  | Train
      { cTrainCSV    :: FilePath
      , cValidRatio  :: Double
      , cEpochs      :: Int
      , cLearningRate:: Double
      }
  | Predict
      { 
        cInputCSV    :: FilePath
      }
  | Eval
      { 
        cTestCSV     :: FilePath
      }
  deriving (Show)

-- | CLI parser entrypoint
parseCLI :: ParserInfo Command
parseCLI = info (hsubparser
    ( command "shell"   (shellCmd   `withInfo` "Run interactive shell")
    <> command "init"    (initCmd    `withInfo` "Initialize a new network")
    <> command "train"   (trainCmd   `withInfo` "Train (stub) on CSV data")
    <> command "predict" (predictCmd `withInfo` "Predict labels for new inputs")
    <> command "eval"    (evalCmd    `withInfo` "Evaluate accuracy on test data")
    ))
  ( fullDesc
  <> progDesc "SimpleNN: neural-net CLI"
  <> header   "snn" )
  where
    withInfo p desc = info (p <**> helper) (progDesc desc)

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
  <$> strOption
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
  <$> strOption
        ( long "input"
        <> metavar "CSV"
        <> help "CSV of new inputs (label column ignored)" )

-- | eval command
evalCmd :: Parser Command
evalCmd =
    Eval
  <$> strOption
        ( long "test"
        <> metavar "CSV"
        <> help "MNIST CSV for testing (label + pixels)" )

-- | Shell command
shellCmd :: Parser Command
shellCmd = pure Shell

-- | Helper to parse Activation from string
parseAct :: ReadM Activation
parseAct = eitherReader $ \s -> case List.map toLower s of
  "relu"     -> Right ReLU
  "softmax"  -> Right Softmax
  "identity" -> Right Identity
  _          -> Left $ "Unknown activation: " ++ s

runCLI :: Command -> IO ()
runCLI Init{..} = do
  net <- initializeNetwork cInputSize cLayerSizes cActivations
  putStrLn "=== Initialized network ==="
  print net
  putStrLn "Note: For interactive use, run 'snn shell' then 'init'."

runCLI Train{} = do
  putStrLn "Error: 'train' command requires network parameters. Please use 'snn shell' and then 'init' followed by 'train'."
  putStrLn "Alternatively, define a standalone 'train' command with full network arguments if desired."

runCLI Predict{} = do
  putStrLn "Error: 'predict' command requires a network. Please use 'snn shell' and then 'init' followed by 'predict'."

runCLI Eval{} = do
  putStrLn "Error: 'eval' command requires a network. Please use 'snn shell' and then 'init' followed by 'eval'."

runCLI Shell = do
  putStrLn "=== SimpleNN Interactive Shell ==="
  putStrLn "Type 'help' for available commands, 'exit' or 'quit' to leave."

  networkRef <- newIORef (Nothing :: Maybe Network)
  shellWorker networkRef
  where
    shellWorker :: IORef (Maybe Network) -> IO ()
    shellWorker networkRef = do
      putStr "> "
      hFlush stdout
      line <- getLine
      let trimmedLine = List.dropWhileEnd (`elem` " \t\n\r") $ List.dropWhile (`elem` " \t") line
      case List.map toLower (List.takeWhile (`notElem` " \t") trimmedLine) of
        ""     -> shellWorker networkRef
        "exit" -> exitSuccess
        "quit" -> exitSuccess
        "help" -> do
            putStrLn "Available commands (use with their flags):"
            putStrLn "  init -i N -l SIZE... -a ACTIVATION... : Initialize a new network"
            putStrLn "  train --train CSV [-r R] [-e N] [--lr RATE] : Train on CSV data"
            putStrLn "  predict --input CSV : Predict labels for new inputs"
            putStrLn "  eval --test CSV : Evaluate accuracy on test data"
            putStrLn "  show : Show current network configuration"
            putStrLn "  exit/quit : Exit the interactive shell"
            putStrLn "  help : Show this help message"
            shellWorker networkRef
        _      -> do
          let args = words trimmedLine
          let parserInfo = parseCLI
          let result = execParserPure (prefs showHelpOnError) parserInfo args

          case result of
            Success cmd -> do
              currentNetM <- readIORef networkRef

              case cmd of
                Init{..} -> do
                  net <- initializeNetwork cInputSize cLayerSizes cActivations
                  writeIORef networkRef (Just net)
                  putStrLn "=== Initialized network ==="
                  shellWorker networkRef

                Train{..} -> do
                  case currentNetM of
                    Just net -> do
                      raw  <- loadCSV cTrainCSV
                      let (trainV, validV) = splitDataset cValidRatio raw
                      putStrLn $ "Loaded " ++ show (length raw) ++ " samples"
                      putStrLn $ " -> training:   " ++ show (length trainV)
                      putStrLn $ " -> validation: " ++ show (length validV)
                      putStrLn "=== (Stub) Training complete ==="
                    Nothing -> putStrLn "Error: No network initialized. Use 'init' command first."
                  shellWorker networkRef

                Predict{..} -> do
                  case currentNetM of
                    Just net -> do
                      raw   <- loadCSV cInputCSV
                      let feats = map (V.toList . V.tail) raw
                          preds = map (predictUnit net) feats
                      putStrLn "=== Predictions ==="
                      mapM_ print preds
                    Nothing -> putStrLn "Error: No network initialized. Use 'init' command first."
                  shellWorker networkRef

                Eval{..} -> do
                  case currentNetM of
                    Just net -> do
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
                    Nothing -> putStrLn "Error: No network initialized. Use 'init' command first."
                  shellWorker networkRef

                Shell{} -> do
                  putStrLn "Already in interactive mode. Type 'exit' to leave."
                  shellWorker networkRef

            Failure failure -> do
              let (msg, _) = renderFailure failure ""
              putStrLn $ "Error: " ++ msg
              shellWorker networkRef
            CompletionInvoked completion -> do
              putStrLn "Completion invoked (not fully supported in this simple shell)."
              shellWorker networkRef