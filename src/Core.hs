{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleContexts #-}

module Core where

import System.Random
import Control.Monad.Random
import Data.List (foldl', transpose, genericLength, foldl1')
import Data.Function ((&))

-----------------------------
-- Custom matrix/vector implementation
-----------------------------

-- Vector as list of Doubles
type Vector = [Double]

-- Matrix as list of rows (each row is a Vector)
type Matrix = [Vector]

-- Create matrix from dimensions and flat list
matrix :: Int -> Int -> [Double] -> Matrix
matrix rows cols vals
  | rows * cols /= length vals = error "Incorrect number of elements for matrix"
  | otherwise = take rows $ chunksOf cols vals

-- Split list into chunks of given size
chunksOf :: Int -> [a] -> [[a]]
chunksOf _ [] = []
chunksOf n xs = take n xs : chunksOf n (drop n xs)

-- Matrix-vector multiplication
matVecMult :: Matrix -> Vector -> Vector
matVecMult mat vec = 
  [ sum [ m * v | (m, v) <- zip row vec ] | row <- mat ]

-- Element-wise vector addition
vecAdd :: Vector -> Vector -> Vector
vecAdd = zipWith (+)

-- Scale vector by scalar
scale :: Double -> Vector -> Vector
scale s = map (s *)

-- Constant vector
konst :: Double -> Int -> Vector
konst k n = replicate n k

-- Apply function to each vector element
cmap :: (Double -> Double) -> Vector -> Vector
cmap = map

-- Sum of vector elements
sumElements :: Vector -> Double
sumElements = sum

-- Matrix element access by index
atIndex :: Matrix -> (Int, Int) -> Double
atIndex m (i, j) = (m !! i) !! j

-- Reshape vector into matrix
reshape :: Int -> Vector -> Matrix
reshape n vec
  | n <= 0 = error "Invalid number of rows"
  | otherwise = chunksOf (length vec `div` n) vec

-- Find index of maximum element
maxIndex :: Vector -> Int
maxIndex = fst . foldl1' maxBy . zip [0..]
  where
    maxBy (i1, v1) (i2, v2)
      | v1 >= v2  = (i1, v1)
      | otherwise = (i2, v2)

-- Generate random vector
randomVector :: RandomGen g => g -> Int -> IO (Vector, g)
randomVector gen len = do
  let (vec, gen') = randomList gen len
  return (vec, gen')
  where
    randomList g n = 
      let (vals, g') = foldl' (\(acc, g) _ -> 
                          let (x, g') = random g
                          in (x:acc, g'))
                      ([], g) [1..n]
      in (vals, g')

-----------------------------
-- Neural network core
-----------------------------

-- Activation functions
data Activation = ReLU | Softmax | Identity deriving (Show, Eq)

-- Neuron representation
data Neuron = Neuron
  { layerIndex :: !Int
  , neuronIndex :: !Int
  , biasValue :: !Double
  , activationFunc :: !Activation
  } deriving (Show)

-- Connection representation
data Connection = Connection
  { fromLayer :: !Int
  , fromNeuron :: !Int
  , toLayer :: !Int
  , toNeuron :: !Int
  , weightValue :: !Double
  } deriving (Show)

-- Neural network layer
data Layer = Layer
  { weightsMatrix :: !Matrix
  , biasesVector :: !Vector
  , layerActivation :: !Activation
  } deriving (Show)

-- Full neural network
data Network = Network
  { inputSize :: !Int
  , layerSizes :: ![Int]
  , layers :: ![Layer]
  } deriving (Show)

-- Initialize a single layer
initializeLayer :: Int -> Int -> Activation -> IO Layer
initializeLayer inputSize outputSize activation = do
  weights <- initializeWeights activation inputSize outputSize
  let biases = konst 0 outputSize
  return $ Layer weights biases activation

-- Initialize weights for a layer
initializeWeights :: Activation -> Int -> Int -> IO Matrix
initializeWeights activation inS outS = do
  let n = fromIntegral inS
      stdDev = case activation of
                 ReLU -> sqrt (2.0 / n)  -- He initialization
                 _    -> sqrt (1.0 / n)  -- Xavier initialization
  
  gen <- newStdGen
  (weights, _) <- randomVector gen (outS * inS)
  let scaledWeights = scale stdDev weights
  return $ reshape outS scaledWeights


-- Initialize full network
initializeNetwork :: Int -> [Int] -> [Activation] -> IO Network
initializeNetwork inS sizes activations 
  | length sizes /= length activations = 
      error "Number of layers must match number of activation functions"
  | otherwise = do
      let prevSizes = inS : init sizes
          layerInputs = zip prevSizes sizes
          configs = zip layerInputs activations  -- Создаем правильные конфигурации
          
      layerList <- mapM (\((prev, next), act) -> initializeLayer prev next act) configs
      
      return $ Network inS sizes layerList

-- Apply activation function to vector
applyActivation :: Activation -> Vector -> Vector
applyActivation ReLU v = map (max 0) v
applyActivation Softmax v = 
  let expV = map exp v
      total = sum expV
  in map (/ total) expV
applyActivation Identity v = v

-- Forward pass through single layer
forwardLayer :: Layer -> Vector -> Vector
forwardLayer Layer{..} input = 
  let z = vecAdd (matVecMult weightsMatrix input) biasesVector
  in applyActivation layerActivation z

-- Forward pass through entire network
forwardNetwork :: Network -> Vector -> Vector
forwardNetwork Network{..} input = 
  foldl' (\inputVec layer -> forwardLayer layer inputVec) input layers

-- Extract neurons from network
extractNeurons :: Network -> [Neuron]
extractNeurons Network{..} = 
  let inputNeurons = [Neuron 0 i 0.0 Identity | i <- [0..inputSize-1]]
      
      hiddenNeurons = concat $ 
        zipWith3 (\layerIdx size layer -> 
          [Neuron (layerIdx+1) j (biasesVector layer !! j) (layerActivation layer)
          | j <- [0..size-1]])
        [1..] layerSizes layers
  in inputNeurons ++ hiddenNeurons
{-
-- Extract connections from network
extractConnections :: Network -> [Connection]
extractConnections Network{..} = 
  let inputConnections = 
        if not (null layers)
          then [Connection 0 i 1 j (atIndex (weightsMatrix (head layers)) (j, i))
               | i <- [0..inputSize-1]
               , j <- [0..head layerSizes - 1]
          else []
      
      hiddenConnections = concat $ 
        zipWith3 (\layerIdx prevSize layer -> 
          [Connection layerIdx i (layerIdx+1) j (atIndex (weightsMatrix layer) (j, i))
          | i <- [0..prevSize-1]
          , j <- [0..layerSizes !! layerIdx - 1]])
        [1..] (init layerSizes) (tail layers)
  in inputConnections ++ hiddenConnections
-}
-- Convert label to one-hot vector
labelToOneHot :: Int -> Int -> Vector
labelToOneHot numClasses lbl = 
  [if i == lbl then 1.0 else 0.0 | i <- [0..numClasses-1]]

-- Predict class from input
predictUnit :: Network -> Vector -> Int
predictUnit network input = 
  maxIndex $ forwardNetwork network input

-- Create XOR dataset
createXORDataset :: [Vector]
createXORDataset = 
  [ [0,0], [0,1], [1,0], [1,1] ]
{-
-- Demonstration function
main :: IO ()
main = do
  putStrLn "Initializing neural network for XOR problem..."
  let config = (2, [2, 1], [ReLU, Identity])  -- 2 inputs, hidden layer of 2, output of 1
  
  network@(Network _ _ layers) <- initializeNetwork (fst3 config) (snd3 config) (trd3 config)
  putStrLn $ "Network initialized with " ++ show (length layers) ++ " layers"
  
  let neurons = extractNeurons network
      connections = extractConnections network
  
  putStrLn "\nNeurons:"
  mapM_ print neurons
  
  putStrLn "\nConnections:"
  mapM_ print connections
  
  putStrLn "\nTesting XOR:"
  mapM_ (\input -> do
      let output = forwardNetwork network input
      putStrLn $ "Input: " ++ show input 
              ++ " -> Output: " ++ show output
    ) createXORDataset

  where
    fst3 (a, _, _) = a
    snd3 (_, b, _) = b
    trd3 (_, _, c) = c
  -}