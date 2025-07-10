{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleContexts #-}

module SimpleNN.Core where

import Numeric.LinearAlgebra hiding ((|>))
import Numeric.LinearAlgebra.Data
import System.Random
import Control.Monad.Random
import Data.List (foldl')
import Data.Function ((&))

-- Data Types
data Activation = ReLU | Softmax | Identity deriving (Show, Eq)

-- Neuron representation (for visualisation)
data Neuron = Neuron
  { layerIndex :: !Int        -- Layer index (0 - input)
  , neuronIndex :: !Int       -- Neuron in layer index
  , biasValue :: !Double      -- Bias
  , activationFunc :: !Activation  -- Activation Func
  } deriving (Show)

-- Representation of connections
data Connection = Connection
  { fromLayer :: !Int         -- source layer
  , fromNeuron :: !Int        -- source neuron
  , toLayer :: !Int           -- destination layer
  , toNeuron :: !Int          -- destination neuron
  , weightValue :: !Double    -- weight
  } deriving (Show)

-- NN layer
data Layer = Layer
  { weightsMatrix :: !(Matrix Double)  -- Weights matrix
  , biasesVector :: !(Vector Double)   -- Vector of biases
  , layerActivation :: !Activation     -- Activation Function
  } deriving (Show)

-- Full Network
data Network = Network
  { inputSize :: !Int          -- Размер входного слоя
  , layerSizes :: ![Int]       -- Размеры скрытых и выходного слоев
  , layers :: ![Layer]         -- Слои сети
  } deriving (Show)

-- 1. Initialize network  ------------------------------------------------------

-- Initialize one layer
initializeLayer :: Int -> Int -> Activation -> IO Layer
initializeLayer inputSize outputSize activation = do
  -- Initialize weights
  weights <- initializeWeights activation inputSize outputSize
  
  -- Zeros
  let biases = konst 0 outputSize
  
  return $ Layer weights biases activation

-- Weight initializarion
initializeWeights :: Activation -> Int -> Int -> IO (Matrix Double)
initializeWeights activation inS outS = do
  let stdDev = case activation of
        ReLU -> sqrt (2.0 / fromIntegral inS)      -- He
        _    -> sqrt (1.0 / fromIntegral inS)      -- Xavier
    
  -- Generation of random weight matrix
  gen <- newStdGen
  let (weights, _) = randomVector gen Gaussian (outS * inS) 
        & scale stdDev
        & reshape outS
  
  return weights

-- Initialize full network
initializeNetwork :: Int -> [Int] -> [Activation] -> IO Network
initializeNetwork inS sizes activations 
  | length sizes /= length activations = 
      error "Number of layers and activation functions must be equal"
  | otherwise = do
      let layerCfgs = zip sizes activations
      layerList <- forM (zip [inS] (init sizes) `zip` layerCfgs) $ 
        \((prevSize, nextSize), (size, act)) -> 
          initializeLayer prevSize nextSize act
      
      return $ Network inS sizes layerList

-- 2. Activation funtions  -------------------------------------------------------

-- Apply activation func to a vector
applyActivation :: Activation -> Vector Double -> Vector Double
applyActivation ReLU = cmap (max 0)
applyActivation Softmax = \v -> 
  let expV = cmap exp v
      total = sumElements expV
  in scale (1/total) expV
applyActivation Identity = id

-- 3. Forward Propogation --------------------------------------------------

-- Propogation through layer
forwardLayer :: Layer -> Vector Double -> Vector Double
forwardLayer Layer{..} input = 
  let z = weightsMatrix #> input + biasesVector
  in applyActivation layerActivation z

-- Propogation through all network
forwardNetwork :: Network -> Vector Double -> Vector Double
forwardNetwork Network{..} input = 
  foldl' (\inputVec layer -> forwardLayer layer inputVec) input layers

-- 4. Network -> Struct for visualization ------------------------

-- List of neurons
extractNeurons :: Network -> [Neuron]
extractNeurons Network{..} = 
  let inputNeurons = [Neuron 0 i 0.0 Identity | i <- [0..inputSize-1]]
      
      -- Hidden and ounput neurons
      hiddenNeurons = concat $ 
        zipWith3 (\layerIdx size layer -> 
          [Neuron (layerIdx+1) j (biasesVector layer ! j) (layerActivation layer)
          | j <- [0..size-1]])
        [1..] layerSizes layers
  in inputNeurons ++ hiddenNeurons

-- Creation of list of connections
extractConnections :: Network -> [Connection]
extractConnections Network{..} = 
  let inputConnections = 
        if not (null layers)
          then [Connection 0 i 1 j (weightsMatrix (head layers) `atIndex` (j, i))
               | i <- [0..inputSize-1]
               , j <- [0..layerSizes !! 0 - 1]]
          else []
      
      hiddenConnections = concat $ 
        zipWith3 (\layerIdx prevSize layer -> 
          [Connection layerIdx i (layerIdx+1) j (weightsMatrix layer `atIndex` (j, i))
          | i <- [0..prevSize-1]
          , j <- [0..layerSizes !! layerIdx - 1]])
        [1..] (init layerSizes) (tail layers)
  in inputConnections ++ hiddenConnections

-- 5. Helper functions -------------------------------------------------

-- Label to One-Hot vector
labelToOneHot :: Int -> Int -> Vector Double
labelToOneHot numClasses lbl = 
  fromList [if i == lbl then 1.0 else 0.0 | i <- [0..numClasses-1]]

-- Predict method
predictUnit :: Network -> Vector Double -> Int
predictUnit network input = 
  maxIndex $ forwardNetwork network input

-- Get test sample of data
createXORDataset :: IO [Vector Double]
createXORDataset = return $ map fromList 
  [[0,0], [0,1], [1,0], [1,1]]

-- 6. Demo function ------------------------------------------------

main :: IO ()
main = do
  putStrLn "Инициализация нейронной сети для XOR..."
  
  -- Конфигурация сети: 2 входа, 2 скрытых нейрона, 1 выход
  let config = (2, [2, 1], [ReLU, Identity])
  
  -- Инициализация сети
  network@(Network _ _ layers) <- initializeNetwork (fst3 config) (snd3 config) (trd3 config)
  putStrLn $ "Сеть инициализирована с " ++ show (length layers) ++ " слоями"
  
  -- Извлечение структуры для визуализации
  let neurons = extractNeurons network
      connections = extractConnections network
  
  putStrLn "\nНейроны:"
  mapM_ print neurons
  
  putStrLn "\nСвязи:"
  mapM_ print connections
  
  -- Тестирование на XOR данных
  dataset <- createXORDataset
  putStrLn "\nТестирование XOR:"
  mapM_ (\input -> do
      let output = forwardNetwork network input
      putStrLn $ "Input: " ++ show (toList input) 
              ++ " -> Output: " ++ show (toList output)
    ) dataset
  where
    fst3 (a, _, _) = a
    snd3 (_, b, _) = b
    trd3 (_, _, c) = c