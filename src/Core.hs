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

-- Derivatives of activation functions
activationDerivative :: Activation -> Vector -> Vector
activationDerivative ReLU = map (\x -> if x > 0 then 1 else 0)  -- Derivative is 1 for positive, 0 otherwise
activationDerivative Identity = konst 1 . length                -- Constant 1 vector
activationDerivative Softmax = error "Softmax derivative not implemented directly"

-- Matrix transpose
transposeMat :: Matrix -> Matrix
transposeMat [] = []
transposeMat ([]:_) = []
transposeMat m = map head m : transposeMat (map tail m)

-- Outer product of two vectors
outer :: Vector -> Vector -> Matrix
outer u v = [[x * y | y <- v] | x <- u]

-- Hadamard product (element-wise multiplication)
hadamard :: Vector -> Vector -> Vector
hadamard = zipWith (*)

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

-- Type for caching intermediate values during forward pass
type LayerCache = (Vector, Vector)  -- (z = W·a_prev + b, a_prev = previous layer's activation)

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

-- Forward pass through single layer with caching
forwardLayer :: Layer -> Vector -> (Vector, LayerCache)
forwardLayer Layer{..} input = 
  let z = vecAdd (matVecMult weightsMatrix input) biasesVector
      a = applyActivation layerActivation z
  in (a, (z, input))

-- Forward pass through entire network with caching
forwardPass :: Network -> Vector -> (Vector, [LayerCache])
forwardPass Network{..} input = 
  let (output, caches) = foldl' 
        (\(a_prev, caches) layer -> 
          let (a, cache) = forwardLayer layer a_prev
          in (a, cache : caches))
        (input, []) layers
  in (output, reverse caches)


-- Backward pass to compute gradients
backwardPass :: Network -> [LayerCache] -> Vector -> [(Matrix, Vector)]
backwardPass net caches target = 
  let layersList = layers net
      numLayers = length layersList
      (lastZ, _) = last caches
      output = applyActivation (layerActivation (last layersList)) lastZ
      
      -- Special handling for output layer
      outputDelta = case layerActivation (last layersList) of
        Softmax -> 
          -- Combined derivative for softmax + cross-entropy
          vecAdd output (scale (-1) target)
        Identity -> 
          -- For regression tasks (mean squared error)
          vecAdd output (scale (-1) target)
        _ -> 
          error "Unsupported output activation function"
      
      -- Compute deltas for each layer
      deltas = 
        if numLayers == 0 
          then []
          else 
            let initDeltas = [outputDelta]
                layerIndices = [numLayers-2, numLayers-3 .. 0]
                computeDelta deltasAcc layerIdx = 
                  let layer = layersList !! layerIdx
                      (z, a_prev) = caches !! layerIdx
                      delta_next = head deltasAcc
                      weightsT = transposeMat (weightsMatrix (layersList !! (layerIdx + 1)))
                      backprop = matVecMult weightsT delta_next
                      deriv = activationDerivative (layerActivation layer) z
                      delta = hadamard backprop deriv
                  in delta : deltasAcc
            in foldl' computeDelta initDeltas layerIndices
      
      -- Compute gradients for each layer
      grads = [ ( outer delta a_prev, delta )  -- (dW, db)
              | i <- [0 .. numLayers - 1]
              , let (z, a_prev) = caches !! i
                    delta = deltas !! i
              ]
  in grads

-- Matrix addition
matrixAdd :: Matrix -> Matrix -> Matrix
matrixAdd = zipWith vecAdd

-- Scale matrix by scalar
scaleMatrix :: Double -> Matrix -> Matrix
scaleMatrix s = map (scale s)

-- Scale gradient by scalar
scaleGrad :: Double -> (Matrix, Vector) -> (Matrix, Vector)
scaleGrad s (dW, db) = (scaleMatrix s dW, scale s db)

-- Add two gradients
addGrads :: (Matrix, Vector) -> (Matrix, Vector) -> (Matrix, Vector)
addGrads (dW1, db1) (dW2, db2) = 
  (matrixAdd dW1 dW2, vecAdd db1 db2)


-- Update network parameters
updateNetwork :: Network -> Double -> [[(Matrix, Vector)]] -> Network
updateNetwork net lr gradsList = 
  let numSamples = genericLength gradsList
      layersGrads = transpose gradsList
      sumGrads = map (foldl1' addGrads) layersGrads
      avgGrads = map (scaleGrad (1 / numSamples)) sumGrads
      newLayers = zipWith 
        (\layer (dW, db) -> 
          layer { weightsMatrix = matrixAdd (weightsMatrix layer) (scaleMatrix (-lr) dW)
                 , biasesVector = vecAdd (biasesVector layer) (scale (-lr) db)
                 })
        (layers net) avgGrads
  in net { layers = newLayers }


-- Train on a single mini-batch
trainOnMiniBatch :: Network 
                 -> Double   -- Learning rate
                 -> [Vector] -- Inputs
                 -> [Vector] -- Targets
                 -> Network
trainOnMiniBatch net lr inputs targets = 
  let gradsList = [ backwardPass net (snd $ forwardPass net input) target 
                 | (input, target) <- zip inputs targets 
                 ]
  in updateNetwork net lr gradsList

-- Create batches for training
createBatches :: Int -> [a] -> [[a]]
createBatches _ [] = []
createBatches size xs = 
  let (batch, rest) = splitAt size xs
  in batch : createBatches size rest

-- Full training procedure
train :: Network 
      -> Double   -- Learning rate
      -> Int      -- Batch size
      -> [Vector] -- Inputs
      -> [Vector] -- Targets
      -> Int      -- Epochs
      -> Network
train net lr batchSize inputs targets epochs = 
  let numExamples = length inputs
      dataset = zip inputs targets
      epochLoop net' epoch = 
        let batches = createBatches batchSize dataset
            batchLoop net'' batch = 
              let (inputs_b, targets_b) = unzip batch
              in trainOnMiniBatch net'' lr inputs_b targets_b
        in foldl' batchLoop net' batches
  in foldl' epochLoop net [1..epochs]

-- Cross-entropy loss function
crossEntropyLoss :: Vector -> Vector -> Double
crossEntropyLoss output target = 
  let losses = zipWith (\o t -> -t * log o) output target
  in sum losses


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

-- | Forward pass without caching (used for predictions)
forwardNetwork :: Network -> Vector -> Vector
forwardNetwork net input = fst (forwardPass net input)

-- | Predict class index from input
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