#JAVA编程语言

## 1. JAVA接口/继承/内部类

**接口与类的区别:**
	
	1.接口不能用于实例化对象。
	2.接口没有构造方法。
	3.接口中所有的方法必须是抽象方法。
	4.接口不能包含成员变量，除了 static 和 final 变量。
	5.接口不是被类继承了，而是要被类实现。
	6.接口支持多继承。

**接口特性**

	1.接口中每一个方法也是隐式抽象的,接口中的方法会被隐式的指定为 public abstract（只能是 public abstract，其他修饰符都会报错）。

	2.接口中可以含有变量，但是接口中的变量会被隐式的指定为 public static final 变量（并且只能是 public，用 private 修饰会报编译错误）。

	3.接口中的方法是不能在接口中实现的，只能由实现接口的类来实现接口中的方法。

	4.一个类实现了某个接口，则必须实现此接口的所有方法[因为接口中的方法全部为abstract类型，abatract方法必须被子类实现]。

**接口和抽象类**

	1. 抽象类中的方法可以有方法体，就是能实现方法的具体功能，但是接口中的方法不行。
	
	2. 抽象类中的成员变量可以是各种类型的，而接口中的成员变量只能是 public static final 类型的。
	
	3. 接口中不能含有静态代码块以及静态方法(用 static 修饰的方法)，而抽象类是可以有静态代码块和静态方法。
	
	4. 一个类只能继承一个抽象类，而一个类却可以实现多个接口。

**继承的特性**

	1.子类拥有父类非private的属性，方法。

	2.子类可以拥有自己的属性和方法，即子类可以对父类进行扩展。

	3.子类可以用自己的方式实现父类的方法。

	4.Java的继承是单继承，但是可以多重继承，单继承就是一个子类只能继承一个父类，多重继承就是，例如A类继承B类，B类继承C类，所以按照关系就是C类是B类的父类，B类是A类的父类，这是java继承区别于C++继承的一个特性。

	5.提高了类之间的耦合性（继承的缺点，耦合度高就会造成代码之间的联系）。

父类中static关键字定义的成员变量，全部只保存一份，而不管其子类有多少个？多个子类共享同一个static 成员变量。

**静态内部类**
	1. 声明在类体部，方法体外，并且使用static修饰的内部类
	2. 脱离外部类的实例独立创建
            在外部类的外部构建内部类的实例
                new Outer.Inner();
            在外部类的内部构建内部类的实例
                new Inner();
	3. 静态内部类体部可以直接访问外部类中所有的静态成员，包含私有。

**成员内部类**
	1.没有使用static修饰的内部类。
	2.在成员内部类中不允许出现*静态变量*和*静态方法*的声明。
	3.成员内部类中可以访问外部类中所有的成员(变量，方法)，包含私有成员。
	4.构建内部类的实例，要求必须外部类的实例先存在.
		外部类的外部/外部类的静态方法：new Outer().new Inner();
		外部类的实例方法：new Inner(); this.new Inner();

**局部内部类**
	1.定义在方法体，甚至比方法体更小的代码块中。
	2.局部内部类可以访问的外部类的成员根据所在方法体不同。
		如果在静态方法中：可以访问外部类中所有静态成员，包含私有
		如果在实例方法中：可以访问外部类中所有的成员，包含私有。
	局部内部类可以访问所在方法中定义的局部变量，但是要求局部变量必须使用final修饰。

**匿名内部类**
	1.没有名字的局部内部类。
	2.没有构造器。
	3.一般隐式的继承某一个父类或者实现某一个接口.

工厂模式的几种简单形态:
	1.简单工厂模式，又称静态工厂方法模式.
	2.工厂方法模式，又称多态性工厂模式.
	3.抽象工厂模式.


## 2. JAVA泛型与反射

### 2.1 深入理解CLASS对象
1. RTTI（Run-Time Type Identification）运行时类型识别。RTTI分为两种：1，编译器即可知道类型的全部信息。2，运行时才知类型的具体信息--反射，对应类为Class类。
	
2. 编译一个新创建的类就会产生一个对应Class对象并且这个Class对象会被保存在同名.class文件里(编译后的字节码文件保存的就是Class对象,也即Class对象表示的是类的类型信息)。
	
3. 当new一个新对象或者引用静态成员变量时，JAVA虚拟机中的类加载子系统会将对应Class对象加载到JVM中，然后JVM再根据这个类型信息相关的Class对象创建我们需要实例对象或者提供静态变量的引用值。注意，对于任何一个类，无论创建多少个实例，JVM中都只有一个与之对应的Class对象--保存整个类的类型信息。Class类只存私有构造函数，因此对应Class对象只能由JVM创建和加载。

<div align=center>
<img src="http://m.qpic.cn/psb?/V14Ifnin2f6pWC/M8AnnOLmggEMMGfe.0jJbAU.2zz6PHSe.58.tFfCs6s!/b/dDEBAAAAAAAA&bo=UAJBAQAAAAADBzA!&rf=viewer_4" width="550" height="250" alt="go struct结构"/>
</div>

4. Class对象的作用在于运行时提供或者获得某个对象的类型信息，对于反射很重要。

Java中共有三种方式获取某个类对应的Class对象 ==》 
	1> Class.forName(ClassName) 返回一个ClassName对应的Class对象的引用; 
	2> Gum_Obj.getClass()方法; 
	3> Class clazz = Gum.class, Class字面常量[无需触发类的最后阶段初始化，应用于借口、数组以及基本数据类型]。
注意调用forName方法时需要捕获一个名称为ClassNotFoundException的异常，因为forName方法在编译器是无法检测到其**传递的字符串对应的类是否存在**的，只能在程序运行时进行检查。
	
	其中实例类的getClass方法和Class类的静态方法forName都将会触发类的初始化阶段，而字面常量获取Class对象的方式则不会触发初始化.
	
	初始化是类加载的最后一个阶段，也就是说完成这个阶段后类也就加载到内存中(Class对象在加载阶段已被创建)，此时可以对类进行各种必要的操作了（如new对象，调用静态成员等），注意在这个阶段，才真正开始执行类中定义的Java程序代码或者字节码。

<div align=center>
<img src="http://m.qpic.cn/psb?/V14Ifnin2f6pWC/1BXm7FhRkFe*G9K2YgZ*rhemgOB9LeuLctFWyDiHSwk!/b/dEQBAAAAAAAA&bo=ugI1AQAAAAADB64!&rf=viewer_4" width="550" height="250" alt="go struct结构"/></div>

在Java中，**所有类型转换都是在运行时进行正确性检查的**,利用RRTI进行判断类型是否正确从而确保强制转换的完成.

**instanceof 关键字与isInstance方法**

instanceof关键字与isInstance方法具有相同的功能 ==》 意在告诉我们对象是不是某个特定的类型[类、接口]实例。

	public void cast2(Object obj){
        //instanceof关键字
        if(obj instanceof Animal){
            Animal animal= (Animal) obj;
        }

        //isInstance方法
        if(Animal.class.isInstance(obj)){
            Animal animal= (Animal) obj;
        	}
  		}
  		
### 2.2 JAVA泛型
泛型类型在逻辑上是多个不同的类型，实际上都是相同的基本类型。泛型提供了编译时类型安全检测机制,而否运行时，该机制允许程序员在编译时检测到非法的类型。**向Class引用添加泛型约束仅仅是为了提供编译期类型的检查从而避免将错误延续到运行时期。**

泛型的使用有三种方式：分别为 泛型类、泛型接口和泛型方法。泛型的类型参数只能是类类型，不能是简单类型。泛型类，是在实例化类的时候指明泛型的具体类型；泛型方法，是在调用方法的时候指明泛型的具体类型 。在Java中，所有的类型转换都是在运行时进行正确性检查的，利用RRTI进行判断类型是否正确从而确保强制转换的完成。

**泛型类**	
	public class GenericsTest {
		public static void main(String[] args) {
		
			Generic<Integer> generic = new Generic<Integer>(1234);
			System.out.println("generic key ==> "+generic.getKey());
		
			Generic<String> generic2 = new Generic<String>("key_value");
			System.out.println("generic2 key ==> "+generic2.getKey());
		}	
	}


	class Generic<T>{
		private T key;
		public Generic(T key){
			this.key = key;
		}
		public T getKey() {
			return key;
		}
	}

 	泛型和Class类的通配符都为<?>，代表所有的类型。
**泛型方法**

	//Generics method
	public static <T> void genericsMethon(Class<T> class1) {
		System.out.println("Class CompleteName ==> "+class1.getCanonicalName());	
	}
	
	public static <T> void toPrintInfo(T... x){
		for(T t:x){
			System.out.print(t+",");
		}
	}

	如果静态方法要使用泛型的话，必须将静态方法也定义成泛型方法 。

**泛型擦除：**


### 2.3 JAVA反射

反射机制是在运行状态中，对于任意一个类，都能够知道这个类的所有属性和方法；对于任意一个对象，都能够调用它的任意一个方法和属性，这种动态获取的信息以及动态调用对象的方法的功能称为java语言的反射机制。在Java中，Class类与java.lang.reflect类库一起对反射技术进行了全力的支持。



#引用

[1. 常见内部类] https://blog.csdn.net/qq_33599978/article/details/70880803
[2. 工厂模式] https://www.jianshu.com/p/bf8341c75304
[3. 深入理解Java类型信息(Class对象)与反射机制] https://blog.csdn.net/javazejian/article/details/70768369


