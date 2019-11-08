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
    let totalEpisode:Int
    let learningRate:Float
    let discountFactor:Float
    let environmentName:String
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
func getAction(actorNet:ActorNet, state:State, hp:Hyperparameters) -> Action{
    let prob = actorNet(state).squeezingShape()
    let actionValue:Int = Int(np.random.choice(hp.actionSize, p:prob.makeNumpyArray()))! //get action by weighted random choice
    let actionProb:Float = Float(prob[actionValue])! //get action's prob
    return Action(value: actionValue, prob: actionProb) //return action
}

func getDoneMask(_ done:Done) -> Tensor<Float>{ //changes done type Done(aka Bool) into doneMask type Tensor<Float>
    if done == false{
        return Tensor<Float>(1)
    }
    else{
        return Tensor<Float>(0)
    }
}

func trainCriticNet(criticNet:CriticNet, criticOptimizer:Adam<CriticNet>, transition:Transition, hp:Hyperparameters) -> (CriticNet, Tensor<Float>){
    var net = criticNet
    let (loss, grads) = net.valueWithGradient {net -> Tensor<Float> in
        let stateValue = net(transition.state)
        let nextStateValue = net(transition.nextState)
        let advantage = (transition.reward + getDoneMask(transition.done) * (stateValue - hp.discountFactor * nextStateValue)).squared().squeezingShape()
        return advantage //get advantage by TD error
    }
    criticOptimizer.update(&net, along:grads)
    return (net, loss)
}

func trainActorNet(actorNet:ActorNet, actorOptimizer:Adam<ActorNet>, transition:Transition, advantage:Tensor<Float>, hp:Hyperparameters) -> (ActorNet, Tensor<Float>){
    var net = actorNet
    let (loss, grads) = net.valueWithGradient {net -> Tensor<Float> in
        let probs = net(transition.state).squeezingShape() //get prob by actor network and squeeze it
        let actionMask = Tensor<Float> (oneHotAtIndices: Tensor<Int32> (Int32(transition.action.value)), depth: hp.actionSize) //get action mask
        let prob = (probs * actionMask).mean() //mask prob by action mask
        return -log(prob)*advantage //return it
    }
    actorOptimizer.update(&net, along:grads)
    return (net, loss)
}

func timeStep(env:PythonObject, actorNet:ActorNet, criticNet:CriticNet, actorOptimizer:Adam<ActorNet>, criticOptimizer:Adam<CriticNet>, previousTransition:Transition, hp:Hyperparameters) -> (ActorNet, CriticNet, Transition){
    let action:Action = getAction(actorNet:actorNet, state:previousTransition.nextState, hp:hp) //decide action
    let (nextState, reward, done, _) = env.step(action.value).tuple4 //do action and get next state, reward, and done
    let transition:Transition = Transition(state: previousTransition.nextState, action: action, reward: reward, nextState: nextState, done: done) //generate transitio
    let (trainedCriticNet, advantage) = trainCriticNet(criticNet: criticNet, criticOptimizer: criticOptimizer, transition: transition, hp: hp) //train critic
    let (trainedActorNet, actorLoss) = trainActorNet(actorNet: actorNet, actorOptimizer: actorOptimizer, transition: transition, advantage: advantage, hp: hp) //train actor
    return (trainedActorNet, trainedCriticNet, transition)
}

func episode(env:PythonObject, actorNet:ActorNet, criticNet:CriticNet, actorOptimizer:Adam<ActorNet>, criticOptimizer:Adam<CriticNet>)
    -> (ActorNet, CriticNet, Reward){
    let initObservation = Tensor<Float> (Tensor<Double> (numpy:env.reset())!)
    var transition = Transition(state: State(0), action: Action(value:0, prob:0), reward: Reward(0), nextState: initObservation, done: Done(0)) //dummy transition
        var score:Float = 0
    var actorNet = actorNet
    var criticNet = criticNet
    //main loop
    while true{
        (actorNet, criticNet, transition) = timeStep(env: env, actorNet: actorNet, criticNet: criticNet, actorOptimizer: actorOptimizer, criticOptimizer: criticOptimizer, previousTransition: transition, hp: hp)
        score += transition.reward
        if transition.done == true{
            break
        }
    }
    return (actorNet, criticNet, score)
}

func main(hp:Hyperparameters){
    var criticNet = CriticNet(stateSize: hp.stateSize, hiddenSize: hp.hiddenSize)
    let criticOptimizer = Adam(for: criticNet, learningRate: hp.learningRate)
    var actorNet = ActorNet(stateSize: hp.stateSize, hiddenSize: hp.hiddenSize, actionSize: hp.actionSize)
    let actorOptimizer = Adam(for: actorNet, learningRate: hp.learningRate)
    let env = gym.make(hp.environmentName)
    var episodeCount:Int = 0
    var score:Reward = 0
    while true{
        (actorNet, criticNet, score) = episode(env: env, actorNet: actorNet, criticNet: criticNet, actorOptimizer: actorOptimizer, criticOptimizer: criticOptimizer)
        episodeCount += 1
        print("episode:\(episodeCount), score:\(score)")
    }
}





