class Account:
    def __init__(self, username, password):
        self.bankname = 'KAIST-IT-Academy' #은행 이름
        self.username = username #사용자 이름
        self.password = password #비밀번호
        self.is_account_number_exist = False #계좌번호 유무
        self.current_balance = 0 # 현재 잔액

    # 계좌번호 생성 (account_number: 계좌번호)
    def create_account_number(self, account_number):
        if self.is_account_number_exist:
            print('기존에 계좌번호가 존재합니다')
        else:
            self.account_number = account_number

    # 입금
    def deposit(self, money):
        self.current_balance += money

    # 출금
    def withdraw(self, money):
        if self.current_balance - money < 0:
            print('출금할 수 없습니다. 계좌 잔액을 확인하세요.')
            return
        self.current_balance -= money

    # 현재 잔고 확인
    def show_current_balance(self):
        print(self.current_balance)

    # 계좌 송금
    def transfer(self, target_account_instance, target_username, target_account_number, money):
        """
        변수 설명
        
        target_account_instance: 송금할 계좌의 인스턴스
        target_username: 송금할 계좌의 사용자 이름
        target_account_number: 송금할 계좌의 계좌번호
        money: 송금할 금액
        """
    
        if (target_account_instance.username != target_username) \
            or (target_account_instance.account_number != target_account_number):
            print('잘못된 정보가 있습니다.')
            return
        if self.current_balance - money < 0:
            print('송금할 수 없습니다. 계좌 잔액을 확인하세요.')
            return
        
        self.current_balance -= money
        target_account_instance.current_balance += money

# 계좌 만들기
account1 = Account('넙죽이', 1234) #넙죽이 계좌
account2 = Account('넙순이', 5678) #넙순이 계좌

# 계좌번호 생성하기
account1.create_account_number(113265)
account2.create_account_number(659889)

# 계좌 입금하기
account1.deposit(10000)

# 계좌 송금하기
account1.transfer(account2, '넙순이', 659889, 3000)

# 계좌 출금하기
account2.withdraw(1000)

# 계좌 잔액 출력하기
account1.show_current_balance()
account2.show_current_balance()