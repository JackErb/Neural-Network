//
//  Matrix Functions.swift
//  Neural Network
//
//  Created by Jack Erb on 12/29/16.
//  Copyright © 2016 Jack Erb. All rights reserved.
//

func -(lhs: [Double], rhs: [Double]) -> [Double] {
    guard lhs.count == rhs.count else {
        fatalError("Matrices were not the same size.")
    }
    
    var result = [Double]()
    for i in 0..<lhs.count {
        result.append(lhs[i] - rhs[i])
    }
    
    return result
}

func -(lhs: [[Double]], rhs: [[Double]]) -> [[Double]] {
    guard lhs.count == rhs.count && lhs[0].count == rhs[0].count else {
        fatalError("Matrices were not the same size.")
    }
    
    var output = [[Double]]()
    for x in 0..<lhs.count {
        output.append(lhs[x] - rhs[x])
    }
    return output
}

func *(lhs: [Double], rhs: [Double]) -> [Double] {
    guard lhs.count == rhs.count else {
        fatalError("Matrices were not the same size.")
    }
    
    var result = [Double]()
    for i in 0..<lhs.count {
        result.append(lhs[i] * rhs[i])
    }
    
    return result
}

func *(lhs: [[Double]], rhs: [[Double]]) -> [[Double]] {
    guard lhs.count == rhs.count && lhs[0].count == rhs[0].count else {
        fatalError("Matrices were not the same size.")
    }
    
    var output = [[Double]]()
    for x in 0..<lhs.count {
        output.append(lhs[x] * rhs[x])
    }
    return output
}

func dot(left: [[Double]], right: [[Double]]) -> [[Double]] {
    guard left[0].count == right.count else {
        fatalError("Matrices are not valid sizes for dot product.")
    }
    
    var output = [[Double]]()
    for i in 0..<left.count {
        output.append([])
        for j in 0..<right[0].count {
            let a1 = left[i]
            let a2 = right.map { $0[j] }
            output[i].append((a1 * a2).reduce(0, +))
        }
    }
    
    return output
}
