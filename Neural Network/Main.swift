//
//  main.swift
//  Neural Network
//
//  Created by Jack Erb on 12/29/16.
//  Copyright © 2016 Jack Erb. All rights reserved.
//

import Foundation

/*    ____x____
     /         \
    i ----x-----o         i: input neuron, x: hidden layer neuron, o: output neuron
     \----x--- /
      \___x___/
 
 */
let layer1 = NeuralLayer(numNeurons: 4, numInputs: 1)
let layer2 = NeuralLayer(numNeurons: 1, numInputs: 4)

let neuralNetwork = NeuralNetwork(layer1: layer1, layer2: layer2)

// Current training process: If number is <= than 0.5, output 0. If it's > 0.5, output 1
for _ in 0..<100000 {
    let num = random(from: 0, to: 1)
    neuralNetwork.train(inputs: [[num]], expectedOutputs: [[num > 0.5 ? 1 : 0]], numInterations: 1)
}

// Expected output should be:   0   1    1     0     0
print(neuralNetwork.calculate([[0],[1],[0.6],[0.4],[0.5]]).layer2)
