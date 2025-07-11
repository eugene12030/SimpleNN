module Main where

import Options.Applicative (execParser)
import CLI                 (parseCLI, runCLI)

main :: IO ()
main = execParser parseCLI >>= runCLI
