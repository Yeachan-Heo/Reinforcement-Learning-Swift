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
func getDoneMask(dones:[Done]) -> [Int]{
    return dones.map({
        (done) -> Int in
        if done == true {
            return 0
        }
        else{
            return 1
        }
    })
}

func getLoss(){
    
}
