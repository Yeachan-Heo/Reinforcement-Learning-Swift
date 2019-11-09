//
//  main.swift
//  PPO
//
//  Created by Yeachan Heo on 09/11/2019.
//  Copyright © 2019 Yeachan Heo. All rights reserved.
//

import Foundation
import TensorFlow
import Python

let np = Python.import("numpy")
let gym = Python.import("gym")

typealias State = Tensor<Float>
typealias Reward = Float
typealias Done = Bool
typealias Action = DiscreteAction

extension Array where Element == Transition {
    func makeBatch() -> ([State], [Action], [Reward], [State], [Done]) {
        let states:[State] = self.map({(transition) -> State in return transition.state})
        let actions:[Action] = self.map({(transition) -> Action in return transition.action})
        let rewards:[Reward] = self.map({(transition) -> Reward in return transition.reward})
        let nextStates:[State] = self.map({(transition) -> State in return transition.nextState})
        let dones:[Done] = self.map({(transition) -> Done in return transition.done})
        return (states, actions, rewards, nextStates, dones)
        
    }
}

struct Transition {
    let state:State
    let action:Action
    let reward:Reward
    let nextState:State
    let done:Done
}

struct DiscreteAction {
    let value:Int
    let prob:Float
}

struct Hyperparameters {
    let stateSize:Int
    let actionSize:Int
    let hiddenSize:Int
}

struct ActorNet : Layer {
    typealias Policy = Tensor<Float>
    var fc1, fc2:Dense<Float>
    
    init(hp:Hyperparameters) {
        fc1 = Dense<Float> (inputSize: hp.stateSize, outputSize: hp.hiddenSize, activation:relu)
        fc2 = Dense<Float> (inputSize: hp.hiddenSize, outputSize: hp.actionSize, activation:softmax)
    }
    
    @differentiable
    func callAsFunction(_ input: State) -> Policy {
        return input.expandingShape(at: 0).sequenced(through: fc1, fc2).squeezingShape()
    }
}

struct CriticNet : Layer {
    typealias Value = Tensor<Float>
    var fc1, fc2:Dense<Float>
    
    init(hp:Hyperparameters) {
        fc1 = Dense<Float> (inputSize: hp.stateSize, outputSize: hp.hiddenSize, activation:relu)
        fc2 = Dense<Float> (inputSize: hp.hiddenSize, outputSize: 1)
    }
    
    @differentiable
    func callAsFunction(_ input: State) -> Value {
        return input.expandingShape(at: 0).sequenced(through: fc1, fc2).squeezingShape()
    }
}

func getAction(actorNet:ActorNet, state:State, hp:Hyperparameters) -> Action {
    let prob:Tensor<Float> = actorNet(state)
    let actionValue:Int = Int(np.random.choice(hp.actionSize, p:prob.makeNumpyArray()))!
    let actionProb = prob[actionValue]
    let action:Action = Action(value: actionValue, prob:Float(actionProb)!)
    return action
}

func trainCritic(criticNet:CriticNet, criticOptimizer:Adam<CriticNet>, transitions:[Transition]) {
    
}






































