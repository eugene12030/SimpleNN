module Main where

import Options.Applicative (execParser)
import qualified CLI as CLI

main :: IO ()
main = execParser CLI.parseCLI >>= CLI.runCLI