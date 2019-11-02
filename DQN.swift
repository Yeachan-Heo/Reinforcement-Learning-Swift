import Foundation
import TensorFlow
import Python

//행렬 연산을 보조하기 위한 넘파이
let np = Python.import("numpy")
//강화학습 환경을 위한 gym
let gym = Python.import("gym")
let env:PythonObject = gym.make("CartPole-v1")

//print(testDiscreteEnvironment(env:env, action:0)) // 환경 테스트하기

//환경 테스트하는 함수
func testDiscreteEnvironment(env:PythonObject, action:Int=0){
    env.reset() //환경 초기화
    let (state, reward, done, info) = env.step(action).tuple4 //액션 하고 상태 받아오기
    print("testing environment\n")
    print("state:\(state), \nreward:\(reward), \ndone:\(done), \ninfo:\(info)\n") //상태 출력하기
    print("tested environment")
}
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

struct Transition{
    let state:State
    let action:DiscreteAction
    let reward:Reward
    let nextState:State
    let done:Bool
}

//////////////////////////funcs//////////////////////////
//랜덤한 Discrete 액션을 선택한다
func getRandomDiscreteAction(_ actionSize:Int) -> DiscreteAction{
    return DiscreteAction(value:Int.random(in: 0 ..< actionSize))
}

// net와 state를 받아 greedy 액션을 선택한다
func getGreedyDiscreteAction(net:DiscreteQNetwork, state:State, stateSize:Int) -> DiscreteAction{
    assert(state.shape == [1,stateSize]) // shape check
    
    let qValue = net(state) // q값 근사하기
    let actionValue = Int(Int32(qValue.argmax(squeezingAxis: 1)[0])!) //실제 행동
    let expectationValue = Float(qValue.max(squeezingAxes: 1)[0]) //기댓값

    return DiscreteAction(value:actionValue , expectation:expectationValue)
}

//epsilon-greedy 정책을 이용해 행동을 선택한다
func getEpsilonGreedyDiscreteAction(net:DiscreteQNetwork, state:State, epsilon:Float, stateSize:Int, actionSize:Int) -> DiscreteAction{
    if Float(drand48()) <= epsilon {
        return getRandomDiscreteAction(actionSize)
    }
    else{
        return getGreedyDiscreteAction(net:net , state:state , stateSize:stateSize )
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

//action들을 가지고 action mask 를 만든다. 이것은 Q값에 곱해서 Loss function에 사용하기 위함이다. (R+rQ(s',a)-Q(s, a))
func getDiscreteActionMask(actions:[DiscreteAction], actionSize:Int) -> Tensor<Float> {
    let actionValues:[Int] = actions.map({(action) -> Int in return action.value})
    let actionMaskIndices:Tensor<Int32> = Tensor<Int32> (actionValues.map({(actionValue) -> Int32 in return Int32(actionValue)}))
    let actionMasks:Tensor<Float> = Tensor<Float> (oneHotAtIndices:actionMaskIndices , depth:actionSize)
    return actionMasks
}

func makeTrainDataSARND(_ transitions:[Transition]) -> ([State], [DiscreteAction], [Reward], [State], [Done]){
    let states:[State] = transitions.map({(transition) -> State in return transition.state})
    let actions:[DiscreteAction] = transitions.map({(transition) -> DiscreteAction in return transition.action})
    let rewards:[Reward] = transitions.map({(transition) -> Reward in return transition.reward})
    let nextStates:[State] = transitions.map({(transition) -> State in return transition.nextState})
    let dones:[Done] = transitions.map({(transition) -> Done in return transition.done})
    return (states, actions, rewards, nextStates, dones)
}

// 메모리에서 샘플링한 트랜지션을 가지고 넷을 학습시킨다음 넷을 리턴한다
func trainNet(mainNet:DiscreteQNetwork, targetNet:DiscreteQNetwork, optimizer:Adam<DiscreteQNetwork> , transitions:[Transition], actionSize:Int, discountFactor:Float) -> DiscreteQNetwork{
    var network = mainNet
    let grads = network.gradient {network -> Tensor<Float> in 
        let (states, actions, rewards, nextStates, dones) = makeTrainDataSARND(transitions)
        let actionMask = getDiscreteActionMask(actions:actions, actionSize:actionSize)
        let doneMask = getDoneMask(dones:dones)
        let qValue = mainNet(Tensor<Float> (states))
        let nextQValue = targetNet(Tensor<Float> (nextStates))
        let maskedQValue = (qValue * actionMask).sum(squeezingAxes:1)
        let maskedNextQValue = nextQValue.max(squeezingAxes:1)
        let loss = Tensor<Float> (rewards) + Tensor<Float> (doneMask) * ((discountFactor * maskedNextQValue) - nextQValue)
        return loss
    } 
    optimizer.update(&network, along:grads)
    return network
}

func timeStep(previousState:State){
    env.step()
}
