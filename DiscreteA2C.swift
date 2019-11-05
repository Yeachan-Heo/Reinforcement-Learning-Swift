//
//  A2C.swift
//  RLimplementation
//
//  Created by Yeachan Heo on 05/11/2019.
//  Copyright © 2019 Yeachan Heo. All rights reserved.
//

import TensorFlow
import Python

//install python dependencies if needed
installPythonPackages(["gym", "numpy"])

//import gym and np
let gym = Python.import("gym")
let np = Python.import("numpy")

// type alias
typealias State = Tensor<Float>
typealias Action = DiscreteAction
typealias Done = Bool
typealias Reward = Float

// structures
struct DiscreteAction{
    var value:Int //액션의 실제값. discrete 한 액션이므로 Int 타입을 가짐
    var prob:Float //액션의 확률값. discrete Actor-Crtic 에서는 주로 softmax policy를 사용하므로 float값을 가짐
}
 
 struct CriticNet : Layer{
     typealias State = Tensor<Float>
     typealias StateValue = Tensor<Float>
     
     var fc1, fc2:Dense<Float>
     
     init(stateSize:Int, hiddenSize:Int){
         fc1 = Dense<Float> (inputSize:stateSize , outputSize:hiddenSize, activation: relu )
         fc2 = Dense<Float> (inputSize:hiddenSize, outputSize:1)
     }

    @differentiable
    func callAsFunction(_ input: State) -> StateValue{
        return input.sequenced(through: fc1, fc2)
     }
 }

struct ActorNet : Layer{
    typealias State = Tensor<Float>
    typealias Policy = Tensor<Float>
    
    var fc1, fc2:Dense<Float>
    
    init(stateSize:Int, hiddenSize:Int, actionSize:Int){
        fc1 = Dense<Float> (inputSize: stateSize, outputSize: hiddenSize, activation: relu)
        fc2 = Dense<Float> (inputSize: hiddenSize, outputSize: actionSize, activation: softmax)
    }
    
    @differentiable
    func callAsFunction(_ input: State) -> Policy {
        return input.sequenced(through: fc1, fc2)
    }
}

struct Hyperparameters{
    let stateSize:Int
    let hiddenSize:Int
    let actionSize:Int
}

func installPythonPackages(_ names:[String]){
    let sp = Python.import("subprocess")
    for name in names{
        sp.call("pip install \(name)", shell:true)
    }
}

func getAction(actorNet:ActorNet, state:State, hp:Hyperparameters) -> Action{
    let prob:Tensor<Float> = actorNet(state).reshaped(to: TensorShape(hp.stateSize))
    let actionValue:Int = Int(np.random.choice(hp.actionSize, p:prob.makeNumpyArray()))!
    let actionProb:Float = Float(prob[actionValue])!
    return Action(value: actionValue, prob: actionProb)
}





