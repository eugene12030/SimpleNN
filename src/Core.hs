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
  | otherwise = take rows (chunksOf cols vals)

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

-- Generate random doubles in [-0.5, 0.5]
randomDoubles :: Int -> IO Vector
randomDoubles n = mapM (\_ -> randomRIO (-0.5, 0.5)) [1..n]

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

-- Neural network layer
data Layer = Layer
  { weightsMatrix :: Matrix
  , biasesVector :: Vector
  , layerActivation :: Activation
  } deriving (Show)

-- Full neural network
data Network = Network
  { inputSize :: Int
  , layerSizes :: [Int]
  , layers :: [Layer]
  } deriving (Show)

-- Type for caching intermediate values during forward pass
type LayerCache = (Vector, Vector)  -- (z = W·a_prev + b, a_prev = previous layer's activation)

-- Initialize a single layer
initializeLayer :: Int -> Int -> Activation -> IO Layer
initializeLayer inS outS activation = do
  weights <- initializeWeights activation inS outS
  let biases = konst 0 outS
  return (Layer weights biases activation)

-- Initialize weights for a layer
initializeWeights :: Activation -> Int -> Int -> IO Matrix
initializeWeights activation inS outS = do
  let n = fromIntegral inS
      stdDev = case activation of
                 ReLU -> sqrt (2.0 / n)
                 _    -> sqrt (1.0 / n)
  ws <- randomDoubles (outS * inS)
  let scaled = scale stdDev ws
  return (reshape inS scaled)


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
      
      return (Network inS sizes layerList)

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
      outputDelta = vecAdd output (scale (-1) target)
      
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
  let gradsList = [ backwardPass net (snd  (forwardPass net input)) target 
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

-- | Forward pass without caching (used for predictions)
forwardNetwork :: Network -> Vector -> Vector
forwardNetwork net input = fst (forwardPass net input)

-- | Predict class index from input
predictUnit :: Network -> Vector -> Int
predictUnit network input = maxIndex (forwardNetwork network input)

-- Create XOR dataset (used for testing of functionality)
createXORDataset :: [Vector]
createXORDataset = 
  [ [0,0], [0,1], [1,0], [1,1] ]