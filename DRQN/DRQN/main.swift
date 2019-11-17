//
//  main.swift
//  DRQN
//
//  Created by Yeachan Heo on 12/11/2019.
//  Copyright © 2019 Yeachan Heo. All rights reserved.
//
import TensorFlow
import Python

let gym:PythonObject = Python.import("gym")
let pil:PythonObject = Python.import("PIL")
let np:PythonObject = Python.import("numpy")

typealias State = Tensor<Float>
typealias Action = Int
typealias Reward = Tensor<Float>
typealias Done = Bool

struct Net : Layer{
    var lstm1:LSTM<Float>
    var dense1:Dense<Float>
    var flatten:Flatten<Float>
    var conv1, conv1b, conv2, conv2b, conv3, conv3b:Conv2D<Float>
    
    init(){
        conv1 = Conv2D<Float>(filterShape: (8,8,1,32), strides: (4,4), activation: relu)
        conv1b = Conv2D<Float>(filterShape: (8,8,32,32), strides: (4,4), activation: relu)
        conv2 = Conv2D<Float>(filterShape: (4,4,32,64), strides: (2,2), activation: relu)
        conv2b = Conv2D<Float>(filterShape: (4,4,64,64), strides: (2,2), activation: relu)
        conv3 = Conv2D<Float>(filterShape: (3,3,64,64), strides: (1,1), activation: relu)
        conv3b = Conv2D<Float>(filterShape: (3,3,64,64), strides: (1,1), activation: relu)
        flatten = Flatten<Float>()
        lstm1 = LSTM<Float>(inputSize:)
        
    }
}

func preProcess(rawObservation:PythonObject) -> Tensor<Float>{
    return Tensor<Float>(numpy:np.array(pil.Image.fromarray(rawObservation).resize([84,84])))!
}
//ㅁㅕㄴㅈㅓㅂ ㄲㅡㅌㄴㅏㅁㅕㄴ ㅁㅏㅁㅜㄹㅣㅎㅏㄲㅖㅅㄷㅏ
