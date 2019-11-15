//
//  main.swift
//  PPO
//
//  Created by Yeachan Heo on 09/11/2019.
//  Copyright © 2019 Yeachan Heo. All rights reserved.


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
    let discountFactor:Float
    let epsilon:Float
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

func getDoneMasks(dones:[Done]) -> Tensor<Float> {
    return Tensor<Float>(dones.map({(done) -> Tensor<Float> in
        if (done == false){
            return Tensor<Float> (1)
        }
        else{
            return Tensor<Float> (0)
        }
    }))
}

func getActionMask(actions:[Action], hp:Hyperparameters) -> Tensor<Float> {
    return Tensor<Float> (oneHotAtIndices: Tensor<Int32>(actions.map({(action) -> Int32 in return Int32(action.value)})), depth: hp.actionSize)
}

func clip(inputs:Tensor<Float>, min:Float, max:Float){
    [0...inputs.shape[0]].map({(idx) -> Tensor<Float> in
        if min(inputs[idx], Tensor<Float>(max)) == Tensor<Float>(max) {
            return inputs[idx] - (inputs[idx] - max)
        }
        else if inputs[idx] < min{
            return inputs[idx] - (inputs[idx] - min)
        }
        else{
            return inputs[idx]
        }
    })
}

func trainCritic(criticNet:CriticNet, criticOptimizer:Adam<CriticNet>, transitions:[Transition], hp:Hyperparameters) -> (CriticNet, Tensor<Float>) {
    var advantage:Tensor<Float> = Tensor<Float>(0)
    var net = criticNet
    
    let grads = net.gradient {(net) -> Tensor<Float> in
        let (states, _, rewards, nextStates, dones) = transitions.makeBatch()
        let values:Tensor<Float> = net(Tensor<Float>(states))
        let nextValues:Tensor<Float> = criticNet(Tensor<Float>(nextStates))
        let target:Tensor<Float> = Tensor<Float> (rewards) + getDoneMasks(dones: dones)*hp.discountFactor*nextValues
        advantage = target - values
        return meanSquaredError(predicted: values, expected: target)
    }
    criticOptimizer.update(&net, along: grads)
    return (net, advantage)
}

func trainActor(actorNet:ActorNet, actorOptimizer:Adam<ActorNet>, advantage:Tensor<Float>, transitions:[Transition], hp:Hyperparameters){
    var net = actorNet
    let grads = net.gradient { (net) -> Tensor<Float> in
        let (states, actions, rewards, nextStates, dones) = transitions.makeBatch()
        let probs:[Float] = actions.map({(action) -> Float in return action.prob})
        let newProbs:Tensor<Float> = net(Tensor<Float>(states))
        let actionMask = getActionMask(actions: actions, hp: hp)
        let ratio = log(exp(newProbs) - exp(Tensor<Float>(probs)))
    }
}






































