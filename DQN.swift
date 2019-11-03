import Foundation
import TensorFlow
import Python

//행렬 연산을 보조하기 위한 넘파이
let np = Python.import("numpy")
//강화학습 환경을 위한 gym
let gym = Python.import("gym")

//////////////////////////typealias//////////////////////////
typealias State = Tensor<Float> 
typealias Reward = Float
typealias Done = Bool

//////////////////////////structs//////////////////////////
//값이 Discrete한 Action을 구조체로 정의
struct DiscreteAction{
    var value:Int // 실제 액션, 액션이 Discrete 하므로 Int 타입을 가진다
    var expectation:Float?
    var prob:Float?
    
    init(value:Int, expectation:Float? = nil, prob:Float? = nil){
        self.value = value
        self.expectation = expectation
        self.prob = prob
    }
}

//2개의 fully-connected layer를 가졌고, State-Action-Value Function(Q(s,a)), aka 큐함수 근사
struct DiscreteQNetwork:Layer{
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>
    
    var fc1, fc2: Dense<Float>
    
    //init에서는 상태, 행동, 은닉층의 크기를 받아 fc1, fc2를 재정의한다
    init(stateSize:Int, actionSize:Int, hiddenSize:Int){
        fc1 = Dense<Float> (inputSize:stateSize, outputSize:hiddenSize, activation:relu)
        fc2 = Dense<Float> (inputSize:hiddenSize, outputSize:actionSize)
    }

    //피드포워드 함수
    @differentiable //<- 미분가능하다
    func callAsFunction(_ input: Input) -> Output {
        return input.sequenced(through:fc1, fc2)
    }
}

struct TransitionMDP{
    let state:State
    let action:DiscreteAction
    let reward:Reward
    let nextState:State
    let done:Bool
}

struct HyperParameters{
    let stateSize:Int
    let hiddenSize:Int
    let actionSize:Int
    let memoryMaxLength:Int
    let sampleSize:Int
    let epsilonDecay:Float
    let discountFactor:Float
}

//////////////////////////extensions//////////////////////////

//for Replay Memory, storeTransitionMDP 함수에서 아마 계속 재할당이 일어날 것이므로 새로은 ReplayMemory Structure를 만들던가 아니면 다른 방법을 시도해야 할 것 같다

extension Array where Element == TransitionMDP{
    func sample(count:Int) -> [TransitionMDP]{
        return Array(self.shuffled()[0 ..< count]) //shuffle 한다음 0~원하는 숫자의 인덱스 까지의 값을 담은 [TransitionMDP] 을 리턴한다
    }

    mutating func storeTransitionMDP(transition:TransitionMDP, hp:HyperParameters){
        self.append(transition)
        if self.count > hp.memoryMaxLength{ // 최대 길이보다 길다면
            self = Array(self[1 ..< self.count]) //하나를 자른다
        }
    }
}

//////////////////////////funcs//////////////////////////
//랜덤한 Discrete 액션을 선택한다
func getRandomDiscreteAction(_ hp:HyperParameters) -> DiscreteAction{
    return DiscreteAction(value:Int.random(in: 0 ..< hp.actionSize))
}

// net와 state를 받아 greedy 액션을 선택한다
func getGreedyDiscreteAction(net:DiscreteQNetwork, state:State, hp:HyperParameters) -> DiscreteAction{
    assert(state.shape == [1,hp.stateSize]) // shape check
    
    let qValue = net(state) // q값 근사하기
    let actionValue = Int(Int32(qValue.argmax(squeezingAxis: 1)[0])!) //실제 행동
    let expectationValue = Float(qValue.max(squeezingAxes: 1)[0]) //기댓값

    return DiscreteAction(value:actionValue , expectation:expectationValue)
}

//epsilon-greedy 정책을 이용해 행동을 선택한다
func getEpsilonGreedyDiscreteAction(net:DiscreteQNetwork, state:State, epsilon:Float, hp:HyperParameters) -> DiscreteAction{
    if Float(drand48()) <= epsilon {
        return getRandomDiscreteAction(hp)
    }
    else{
        return getGreedyDiscreteAction(net:net , state:state.reshaped(to:[1,hp.stateSize]) , hp:hp)
    }
}

// Bool 타입을 가진 Done을 mask로 바꿔준다
func getDoneMask(dones:[Done]) -> [Float]{
    return dones.map({
        (done) -> Float in
        if done == true {
            return 0
        }
        else{
            return 1
        }
    })
}

func generateDummyTransitionMDP(state:State = State(0), action:DiscreteAction = DiscreteAction(value:1), reward:Reward = Reward(0), nextState:State = State(0), done:Done = Done(false)) -> TransitionMDP{
    let dummy = TransitionMDP(state:state, action:action, reward:reward, nextState:nextState, done:done)
    //print(dummy)
    return dummy
}

//action들을 가지고 action mask 를 만든다. 이것은 Q값에 곱해서 Loss function에 사용하기 위함이다. (R+rQ(s',a)-Q(s, a))
func getDiscreteActionMask(actions:[DiscreteAction], hp:HyperParameters) -> Tensor<Float> {
    let actionValues:[Int] = actions.map({(action) -> Int in return action.value})
    let actionMaskIndices:Tensor<Int32> = Tensor<Int32> (actionValues.map({(actionValue) -> Int32 in return Int32(actionValue)}))
    let actionMasks:Tensor<Float> = Tensor<Float> (oneHotAtIndices:actionMaskIndices , depth:hp.actionSize)
    return actionMasks
}

func makeTrainDataSARND(_ transitions:[TransitionMDP]) -> ([State], [DiscreteAction], [Reward], [State], [Done]){
    let states:[State] = transitions.map({(transition) -> State in return transition.state})
    let actions:[DiscreteAction] = transitions.map({(transition) -> DiscreteAction in return transition.action})
    let rewards:[Reward] = transitions.map({(transition) -> Reward in return transition.reward})
    let nextStates:[State] = transitions.map({(transition) -> State in return transition.nextState})
    let dones:[Done] = transitions.map({(transition) -> Done in return transition.done})
    return (states, actions, rewards, nextStates, dones)
}

// 메모리에서 샘플링한 트랜지션을 가지고 넷을 학습시킨다음 넷을 리턴한다
func trainNet(mainNet:DiscreteQNetwork, targetNet:DiscreteQNetwork, optimizer:Adam<DiscreteQNetwork> , samples:[TransitionMDP], hp:HyperParameters) -> DiscreteQNetwork{
    var network = mainNet
    let grads = network.gradient {network -> Tensor<Float> in 
        let (states, actions, rewards, nextStates, dones) = makeTrainDataSARND(samples) //샘플을 s,a,r,s',d 로 바꿔준다
        let actionMask = getDiscreteActionMask(actions:actions, hp:hp) // 액션 마스크를 준비한다(실제로 한 액션의 Q값만 사용하기 위함)
        let doneMask = getDoneMask(dones:dones) // done 마스크를 준비한다(만약 t==T 라면 뒤의 기댓값을 고려하지 않고 reward만 고려한다)
        let qValue = mainNet(Tensor<Float> (states)) //s에 대한 Q값을 구한다
        let nextQValue = targetNet(Tensor<Float> (nextStates)) //s'에 대한 Q값을 구한다
        let maskedQValue = (qValue * actionMask).sum(squeezingAxes:1) //s에 대한 Q값을 actionMask를 이용해 마스킹한다
        let maskedNextQValue = nextQValue.max(squeezingAxes:1) // s'에 대한 Q값의 max 값을 사용한다 
        let loss = Tensor<Float> (rewards) + Tensor<Float> (doneMask) * ((hp.discountFactor * maskedNextQValue) - maskedQValue) //DQN Temporal-Difference Loss
        return loss
    } 
    optimizer.update(&network, along:grads)
    return network
}

func timeStep(env:PythonObject, net:DiscreteQNetwork, targetNet:DiscreteQNetwork, optimizer:Adam<DiscreteQNetwork> , previousTransitionMDP:TransitionMDP, memory:[TransitionMDP], epsilon:Float, hp:HyperParameters)
    -> (DiscreteQNetwork, [TransitionMDP], TransitionMDP, Float){
    let action:DiscreteAction = getEpsilonGreedyDiscreteAction (net:net , state:previousTransitionMDP.nextState , epsilon:epsilon , hp:hp)
    let (nextState, reward, done, _) = env.step(action.value).tuple4
    //print(nextState)
    print(previousTransitionMDP.nextState)
    var  trainedNet:DiscreteQNetwork = net 
    if memory.count > hp.sampleSize{
        let samples:[TransitionMDP] = memory.sample(count: hp.sampleSize)
        
        //for sample in samples{
        //    print(sample)
        //}
        
        trainedNet = trainNet(mainNet:net , targetNet:targetNet , optimizer:optimizer , samples:samples , hp:hp)
    }
    //print(previousTransitionMDP)
    let transition:TransitionMDP = TransitionMDP(state:previousTransitionMDP.nextState , action:action , reward:Reward(reward)! , nextState:State(Tensor<Double>(numpy: nextState)!) , done:Done(done)!)
    var memory = memory
    memory.storeTransitionMDP(transition:transition , hp: hp)
    let epsilon = epsilon * hp.epsilonDecay
    return (trainedNet, memory, transition, epsilon)
}

func episode(env:PythonObject, net:DiscreteQNetwork, targetNet:DiscreteQNetwork, optimizer:Adam<DiscreteQNetwork> , memory:[TransitionMDP], epsilon:Float, hp:HyperParameters)
    -> ([TransitionMDP], DiscreteQNetwork, Reward, Float){
    let pythonInitState = env.reset()
    pythonInitState.dtype = np.float32
    var transition = generateDummyTransitionMDP(nextState:State(numpy:pythonInitState)!)
    var memory = memory
    var mainNet = net
    let targetNet = targetNet
    var score:Reward = 0
    var epsilon = epsilon
    while true{
        (mainNet, memory, transition, epsilon) = timeStep(env:env , net:mainNet , targetNet:targetNet , optimizer:optimizer , previousTransitionMDP:transition , memory:memory , epsilon:epsilon , hp: hp)
        score += transition.reward
        if transition.done == true{
            break
        }
    }
    return (memory, mainNet, score, epsilon)
}

func main(hp:HyperParameters){
    let env = gym.make("CartPole-v1")
    var net = DiscreteQNetwork(stateSize:Int(env.observation_space.shape[0])! , actionSize:Int(env.action_space.n)! , hiddenSize:32 )
    let optimizer = Adam(for: net, learningRate: 0.0001)
    var targetNet = net
    var cnt = 0
    var memory:[TransitionMDP] = []
    var score:Reward = 0
    var epsilon:Float = 1
    while true {
        cnt += 1
        (memory, net, score, epsilon) = episode(env:env, net:net, targetNet:targetNet, optimizer:optimizer, memory: memory, epsilon: epsilon, hp: hp)
        print("episode:\(cnt), score:\(score)")
        if cnt % 2 == 0{
            targetNet = net
        }
        if cnt == 300{
            break
        }
    }

}

let hp = HyperParameters(stateSize:4 , hiddenSize:32 , actionSize:2 , memoryMaxLength:10000 , sampleSize:16 , epsilonDecay:0.999 , discountFactor:0.99 )
main(hp: hp)
