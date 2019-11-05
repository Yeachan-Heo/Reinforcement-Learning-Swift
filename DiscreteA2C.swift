//
//  A2C.swift
//  RLimplementation
//
//  Created by Yeachan Heo on 05/11/2019.
//  Copyright © 2019 Yeachan Heo. All rights reserved.
//

import TensorFlow
import Python

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
 
//critic network, 상태를 입력으로 받아 상태의 가치를 근사한다
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

//actor network, 상태를 입력으로 받아 정책을 근사한다. 여기서는 discrete한 action space에 알맞게 softmax 정책을 근사한다.
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

//hyperparameters
struct Hyperparameters{
    let stateSize:Int
    let hiddenSize:Int
    let actionSize:Int
}

//transitions: SARSD
struct Transition{
    let state: State
    let action: Action
    let reward: Reward
    let nextState: State
    let done: Done
}

//functions
func getAction(prob:Tensor<Float>, hp:Hyperparameters) -> Action{
    let actionValue:Int = Int(np.random.choice(hp.actionSize, p:prob.makeNumpyArray()))! //get action by weighted random choice
    let actionProb:Float = Float(prob[actionValue])! //get action's prob
    return Action(value: actionValue, prob: actionProb) //return action
}

func getDoneMask(_ done:Done) -> Tensor<Float>{
    if done == false{
        return Tensor<Float>(1)
    }
    else{
        return Tensor<Float>(0)
    }
}

func getAdvantage(transition:Transition, stateValue:Tensor<Float>, nextStateValue:Tensor<Float>) -> Tensor<Float> {
    let advantage = (transition.reward+getDoneMask(transition.done) * (stateValue-nextStateValue)).squared().squeezingShape() //get advantage by TD error
    return advantage
}

func getActorLoss(transition:Transition, advantage:Tensor<Float>) -> Tensor<Float> {
    return log(transition.action.prob)*advantage
}

func trainCriticNet(criticNet:CriticNet, criticOptimizer:Adam<CriticNet>, transition:Transition) -> (CriticNet, Tensor<Float>){
    var net = criticNet
    let (loss, grads) = net.valueWithGradient {net -> Tensor<Float> in
        let stateValue = net(transition.state)
        let nextStateValue = net(transition.nextState)
        return getAdvantage(transition: transition, stateValue: stateValue, nextStateValue: nextStateValue)
    }
    criticOptimizer.update(&net, along:grads)
    return (net, loss)
}

//derivative 하게 만들기 위해 trainactornet 안에 구현해야 할 것: action 정하고 action 하기, transition(SARSD) 만들어 모델 업데이트...
func trainActorNet(actorNet:ActorNet, actorOptimizer:Adam<ActorNet>, transition:Transition, advantage:Tensor<Float>, hp:Hyperparameters) -> (ActorNet, Tensor<Float>){
    var net = actorNet
    var action:Action = Action(Value:-1, prob:-1)
    let (loss, grads) = net.valueWithGradient {net -> Tensor<Float> in
        let action = getAction(actorNet: actorNet, state: transition.state, hp: hp)
        
    }
}





