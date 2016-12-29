//
//  Network.swift
//  Neural Network
//
//  Created by Jack Erb on 12/29/16.
//  Copyright © 2016 Jack Erb. All rights reserved.
//

import Foundation

//
//  Neural Network.swift
//  Evolution
//
//  Created by Jack Erb on 12/28/16.
//  Copyright © 2016 Jack Erb. All rights reserved.
//

import Foundation

func random(from low: Double, to high: Double) -> Double {
    return Double(arc4random_uniform(UInt32.max)) / Double(UInt32.max) * (high - low) + low
}

class NeuralLayer {
    var synapticWeights = [[Double]]()
    
    // numNeurons: number of neurons in the current layer
    // numInputs : number of inputs into the current neurons, i.e. numbers of neurons from the last layer
    init(numNeurons: Int, numInputs: Int) {
        for i in 0..<numInputs {
            synapticWeights.append([])
            for _ in 0..<numNeurons {
                synapticWeights[i].append(random(from: -1, to: 1))
            }
        }
    }
}


class NeuralNetwork {
    let layer1, layer2: NeuralLayer
    
    init(layer1: NeuralLayer, layer2: NeuralLayer) {
        self.layer1 = layer1
        self.layer2 = layer2
    }
    
    func sigmoid(_ x: Double) -> Double {
        return 1 / (1 + pow(M_E, -x))
    }
    
    func sigmoidDerivative(_ x: Double) -> Double {
        return x * (1 - x)
    }
    
    func train(inputs: [[Double]], expectedOutputs: [[Double]], numInterations: Int) {
        for _ in 0..<numInterations {
            let output = calculate(inputs)
            
            let layerTwoError = expectedOutputs - output.layer2
            let layerTwoDelta = layerTwoError * output.layer2.map { $0.map { sigmoidDerivative($0) } }
            
            
            
            let layerTwoSynapticWeights = layer2.synapticWeights.flatMap { $0 }
            
            let layerOneError = layerTwoDelta.map { (delta: [Double]) -> [Double] in
                var output = [Double]()
                for synapticWeight in layerTwoSynapticWeights {
                    output.append(delta[0] * synapticWeight)
                }
                
                return output
            }
            let layerOneDelta = layerOneError * output.layer1.map { $0.map { sigmoidDerivative($0) } }
            
            
            var trainingInputs = [[Double]]()
            for y in 0..<inputs[0].count {
                trainingInputs.append([])
                for x in 0..<inputs.count {
                    trainingInputs[trainingInputs.count-1].append(inputs[x][y])
                }
            }
            
            let layerOneAdjustment = dot(left: trainingInputs, right: layerOneDelta)
            
            var layerOneOutputs = [[Double]]()
            for y in 0..<output.layer1[0].count {
                layerOneOutputs.append([])
                for x in 0..<output.layer1.count {
                    layerOneOutputs[layerOneOutputs.count-1].append(output.layer1[x][y])
                }
            }
            let layerTwoAdjustment = dot(left: layerOneOutputs, right: layerTwoDelta)
            
            for x in 0..<layer1.synapticWeights.count {
                for y in 0..<layer1.synapticWeights[x].count {
                    layer1.synapticWeights[x][y] += layerOneAdjustment[x][y]
                }
            }
            
            for x in 0..<layer2.synapticWeights.count {
                for y in 0..<layer2.synapticWeights[x].count {
                    layer2.synapticWeights[x][y] += layerTwoAdjustment[x][y]
                }
            }
        }
    }
    
    func calculate(_ inputs: [[Double]]) -> (layer1: [[Double]], layer2: [[Double]]) {
        
        func weight(input: [[Double]], synapticWeights: [[Double]]) -> [[Double]] {
            var output = [[Double]]()
            for i in 0..<input.count {
                output.append([])
                for j in 0..<synapticWeights[0].count {
                    let weights = synapticWeights.map { $0[j] }
                    let neuronOutputs = weights * input[i]
                    
                    output[i].append(neuronOutputs.reduce(0, +))
                }
            }
            return output
        }
        
        let layerOneOutput = weight(input: inputs, synapticWeights: layer1.synapticWeights).map { $0.map { sigmoid($0) } }
        let layerTwoOutput = weight(input: layerOneOutput, synapticWeights: layer2.synapticWeights).map { $0.map { sigmoid($0) } }
        
        return (layerOneOutput, layerTwoOutput)
    }
}
