import sys
sys.path.append("/Users/admin/data/test_project/AlgorithmMarkdown")
from logger import Log_debug
from collections import Counter,defaultdict,deque

class Solution:
    #二分 - D天内运达，最低运载（每次至少一个包裹，左边界包裹的最大值。右边界一次把所有包裹运走）
    def shipWithinDays(self, weights, D):
        left = max(weights)
        right = sum(weights)
        while left < right:
            mid = (left + right) // 2
            #times运载次数，tmp每次的运载量
            times, tmp = 1, 0
            for i in weights:
                if tmp + i > mid:
                    times += 1
                    tmp = 0
                tmp += i
                #判断天数
                # if times > D:
                #     Log_debug.info(f'i : {i}')
                #     break
            #小于D天运载可以压缩，大于D天运载需要提升，
            if times <= D:
                right = mid
            else:
                left = mid + 1

        return left

    #插入位置。插入的位置第一个大于等于target的位置
    def searchInsert(self, nums, target):
        #注意起始值和终值。
        left, right = 0, len(nums) - 1
        #【4，6】之间插，1、mid=4 left后移。2、left=right mid=6 right迁移。3、left>rihgt结束循环
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            if nums[mid] > target:
                right = mid - 1
            else:
                left = mid + 1
        return left

    #
    def minEatingSpeed(self, piles, h):
        def cost(k):
            t = 0
            #为什么是k-1。当数量小于k的时候，加（k-1）可以增加1
            for i in piles:
                t += (i + k - 1) // k
            return t
        #没有采用起始位为0的情况。判断条件小于。相邻情况mid=left，判断条件终right=mid，结束下一轮循环
        left, right = 1, max(piles)
        while left < right:
            mid = (left + right) // 2
            ret = cost(mid)
            if ret <= h:
                right = mid
            else:
                left = mid + 1
        return left



    #原数组是单调的。前后是两条单调数组的临界点。
    def search(self, nums, target):
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            #left与right临近，left==mid。前半段数据
            elif nums[left] <= nums[mid]:
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            #后半段数据
            else:
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
        return -1


    #爬楼梯。n+1可以取到n层的结果。b从1层到2层。a从0层到2层。
    def climbStairs(self, n):
        a = b = 1
        for i in range(2, n + 1):
            a, b = b, a + b
        return b

    #节约了空间，当前被打劫的最大收益只与前两个房间的最大收益相关。
    def rob(self, nums):
        length = len(nums)
        if length == 0:
            return 0
        if length == 1:
            return nums[0]
        left, right = nums[0],max(nums[0], nums[1])
        for i in range(2, length):
            left,right = right,max(left + nums[i], right)
        return right

    #首尾相连情况
    def rob_1(self, nums):
        def new_rob(start, end):
            left, right = nums[start], max(nums[start], nums[start + 1])
            for i in range(start + 2, end ):
                left, right = right, max(left + nums[i], right)
            return right

        length = len(nums)
        if length <= 2:
            return max(nums)
        #从0开始到倒数第一位，从1开始到最后一位
        return max(new_rob(0, length - 1), new_rob(1, length))

    #这个应该是双指针
    def threeSumClosest(self, nums, target):
        #初始化一个无穷大的数
        ret = float('inf')
        nums.sort()
        length = len(nums)
        for i in range(length - 2):
            left = i + 1
            right = length - 1
            while left < right:
                tmp = nums[i] + nums[left] + nums[right]
                ret = tmp if abs(tmp - target) < abs(ret - target) else ret
                if tmp == target:
                    return target
                if tmp > target:
                    right -= 1
                else:
                    left += 1
        return ret


    def subarraySum(self, nums, k):
        '''
        1. 假设数组长度位l，想寻找和为k的数组时，前x个数字的和为10，前y个数字的和为10 + k.
        2. 我们找到了一组和解，然后拆分 y - x = x + z - x = 10 + k - 10 = k。
        '''
        #数组中没有0。这个起到flag的作用，pre_sum==k时，加1。在dic中找是否存在前缀和为X的个数
        dic = Counter({0: 1})
        ret = pre_sum = 0
        for i in nums:
            pre_sum += i
            ret += dic[pre_sum - k]
            dic[pre_sum] += 1
            Log_debug.logger.info(f'dic - {i} : {dic} , {ret} , {pre_sum}')

        return ret

    #不包括中心坐标数据。剔除中心坐标，左右总和相等
    def pivotIndex(self, nums):
        total = sum(nums)
        pre_sum = 0
        for i in range(len(nums)):
            if total - nums[i] == pre_sum * 2:
                return i
            pre_sum += nums[i]
        return -1


    def corpFlightBookings(self, bookings, n):
        #初始化，标记结束位置会超出n的长度。定义n+1长度。
        f = [0] * (n + 1)
        #起始的位置加上seat，结束的位置-seat正好抵消。其他位置上是0
        for first, end, seat in bookings:
            f[first] += seat
            if end < n:
                f[end + 1] -= seat
        for i in range(1, len(f)):
            f[i] = f[i - 1] + f[i]
        return f[1:]

    # gain[i] 是点 i 和点 i + 1 的 净海拔高度差。如何计算i点的净海拔，前缀和即当前点的净海拔
    def largestAltitude(self, gain) -> int:
        ret = pre_sum = 0
        for i in gain:
            pre_sum += i
            ret = max(ret,pre_sum)
        return ret



    def waysToPartition(self, nums , k ) -> int:
        ret = 0
        li = [nums[0]]
        #前缀和
        for i in range(1, len(nums)):
            li.append(li[-1] + nums[i])
        d1 = Counter(li[:-1])
        final = li[-1]
        Log_debug.logger.info(f'd1 : {d1} , final : {final}')
        #找到d1中和为一半儿的位置
        if final % 2 == 0:
            ret = d1.get(final // 2, 0)
        d2 = defaultdict(int)
        #调整k，如果调整后也符合条件，比较ret大小
        for i in range(len(nums)):
            #移除i位置的状态
            d1[li[i]] -= 1
            #替换k
            div = li[i] - nums[i] + k
            #更新新的i位置前缀和。分割位置和替换位置不一定在一起
            d2[div] += 1
            #i位置后的前缀和均变化
            tmp = final - nums[i] + k
            Log_debug.logger.info(f'd1-remove-{i} : {d1} , tmp : {tmp}')
            if tmp % 2 == 1:
                d2[li[i]] += 1
                d2[div] -= 1
                Log_debug.logger.info(f'd2-tmp不符合条件-{i} : {d2}')
                continue
            #d1.get(tmp // 2 - k + nums[i], 0)d1中找是否包含这样的分割点。
            ret = max(ret, d1.get(tmp // 2 - k + nums[i], 0) + d2.get(tmp // 2, 0))
            #
            d2[li[i]] += 1
            d2[div] -= 1
        return ret



    def twoSum(self, numbers, target):
        left, right = 0, len(numbers) - 1
        while left < right:
            if numbers[left] + numbers[right] == target:
                return [left, right]
            elif numbers[left] + numbers[right] > target:
                right -= 1
            else:
                left += 1


    def threeSum(self, nums):
        nums.sort()
        lg, ret = len(nums), []
        for i in range(lg - 2):
            #第一位大于0不符合条件，退出
            if nums[i] > 0:
                break
            #i循环相等的情况跳跃过
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            left = i + 1
            right = lg - 1
            while left < right:
                total = nums[i] + nums[left] + nums[right]
                if total == 0:
                    ret.append([nums[i], nums[left], nums[right]])
                    val = nums[left]
                    #left循环相等的情况跳过
                    while left < right and nums[left] == val:
                        left += 1
                elif total > 0:
                    right -= 1
                else:
                    left += 1
        return ret



    def minSubArrayLen(self, target, nums):
        left = total = 0
        ret = float('inf')
        for right, num in enumerate(nums):
            #total的和是从left开始。达到条件后left会逐渐增加
            total += num
            #当满足条件时，从left逐步减，知道小于target，计算需要的最短数组
            while total >= target:
                #达到条件ret记录该次的数组长度
                ret = min(ret, right - left + 1)
                total -= nums[left]
                left += 1
        #ret > len(nums) 前缀和都小于target
        return 0 if ret > len(nums) else ret


    #与和是一样的逻辑，改成乘法
    def numSubarrayProductLessThanK(self, nums, k):
        left = ret = 0
        total = 1
        for right, num in enumerate(nums):
            total *= num
            while left <= right and total >= k:
                total //= nums[left]
                left += 1
            if left <= right:
                ret += right - left + 1
        return ret



    def subarraySum(self, nums, k):
        ret = pre_sum = 0
        #前缀和放入字典
        pre_dict = {0: 1}
        for i in nums:
            pre_sum += i
            #ret存量（其他位置上） 加 i位置上取到的符合条件的个数
            ret += pre_dict.get(pre_sum - k, 0)
            #标记前缀和有多少个
            pre_dict[pre_sum] = pre_dict.get(pre_sum, 0) + 1
        return ret



    def findMaxLength(self, nums) -> int:
        pre_dict = {0: -1}
        ret = pre_sum = 0
        for index, num in enumerate(nums):
            #方便处理加。相等的情况加合为0
            pre_sum += -1 if num == 0 else 1
            #不在pre_dict中才把pre_sum的索引地址放入
            if pre_sum in pre_dict:
                ret = max(ret, index - pre_dict[pre_sum])
            else:
                pre_dict[pre_sum] = index
        return ret




    def checkInclusion(self, s1, s2) -> bool:
        arr1, arr2, lg = [0] * 26, [0] * 26, len(s1)
        if lg > len(s2):
            return False
        #记录s1中每个单词出现的次数
        for i in range(lg):
            arr1[ord(s1[i]) - ord('a')] += 1
            arr2[ord(s2[i]) - ord('a')] += 1

        for j in range(lg, len(s2)):
            if arr1 == arr2:
                return True
            #S2的起始位置要调整为【j-lg】
            arr2[ord(s2[j - lg]) - ord('a')] -= 1
            arr2[ord(s2[j]) - ord('a')] += 1
        return arr1 == arr2




    def findAnagrams(self, s: str, p: str) :
        arr1, arr2, lg, ret = [0] * 26, [0] * 26, len(p), []
        #特殊情况
        if lg > len(s):
            return []
        for i in range(lg):
            arr1[ord(p[i]) - ord('a')] += 1
            arr2[ord(s[i]) - ord('a')] += 1
        if arr1 == arr2:
            ret.append(0)
        for i in range(lg,len(s)):
            arr2[ord(s[i]) - ord('a')] += 1
            arr2[ord(s[i - lg]) - ord('a')] -= 1
            if arr1 == arr2:
                #相等的时候记录起始索引
                ret.append(i - lg + 1)
        return ret


    def lengthOfLongestSubstring(self, s):
        calc = {}
        left = 0
        ret = 0
        for i, j in enumerate(s):
            if j in calc:
                # 如果重复的数字出现在l之前忽略，否则了跳到该值的下一个位置
                left = max(left, calc[j] + 1)
            calc[j] = i
            ret = max(ret, i - left + 1)
        return ret



    def isPalindrome(self, s):
        left, right, flag = 0, len(s) - 1, False
        while left <= right:
            if not s[left].isalnum():
                left += 1
            #right > left这个条件应该不需要
            elif not s[right].isalnum() and right > left:
                right -= 1
            else:
                if s[left].lower() != s[right].lower():
                    return False
                left += 1
                right -= 1
        return True



    def validPalindrome(self, s):
        def check(l, r):
            while l <= r:
                if s[l] != s[r]:
                    break
                l += 1
                r -= 1
            return l, r

        mid = len(s) // 2
        left, right = check(0, len(s) - 1)
        #返回left大于mid说明左右都相等
        if left > mid:
            return True
        #左边删除或者右边删除
        return check(left + 1, right)[0] > mid or check(left, right - 1)[0] == mid


    def removeNthFromEnd(self, head, n):
        left = right = head
        count = 0
        while count < n:
            right = right.next
            count += 1
        if not right:
            return head.next
        while right.next:
            left = left.next
            right = right.next
        left.next = left.next.next
        return head


    def isAnagram(self, s: str, t: str) -> bool:
        Log_debug.logger.info(f'counter : {Counter(s)}')
        return s != t and Counter(s) == Counter(t)



    def reverseList(self, head):
        pre, cur = None, head
        while cur:
            tmp = cur.next
            cur.next = pre
            pre, cur = cur, tmp
        return pre

    def addTwoNumbers(self, l1, l2):
        rev_l1 = self.reverseList(l1)
        rev_l2 = self.reverseList(l2)
        count = 0
        ret = ListNode()
        #tmp移动结点，ret返回结点
        tmp = ret
        #l1,l2,count均为0退出循环
        while rev_l1 or rev_l2 or count:
            num = 0
            if rev_l1:
                num += rev_l1.val
                rev_l1 = rev_l1.next
            if rev_l2:
                num += rev_l2.val
                rev_l2 = rev_l2.next
            count, num = divmod(num + count, 10)
            tmp.next = ListNode(num)
            tmp = tmp.next
        return self.reverseList(ret.next)


    def asteroidCollision(self, asteroids):
        s, p = [], 0
        while p < len(asteroids):
            if not s or s[-1] < 0 or asteroids[p] > 0:
                s.append(asteroids[p])
            elif s[-1] <= -asteroids[p]:
                #s一定会出队列。asteroids【p】相等进入跳过p+1进入下次循环。否则继续循环p
                if s.pop() < -asteroids[p]:
                    continue
            p += 1
        return s


    def dailyTemperatures(self, temperatures):
        stack, ret = [], [0] * len(temperatures)
        for i, num in enumerate(temperatures):
            #循环弹栈，栈小于num的均被弹出。
            while stack and temperatures[stack[-1]] < num:
                index = stack.pop()
                ret[index] = i - index
            #栈内记录索引
            stack.append(i)
        return ret





    def largestValues(self, root):
        ret, queue = [], deque()
        if root:
            queue.append(root)
        #弹出上一层，队列中会放入下一层的结点。可以判断持续多少层
        while queue:
            num = -float('inf')
            #计算每一层的个数。每一个都会被弹出
            for i in range(len(queue)):
                q = queue.popleft()
                num = max(num, q.val)
                if q.left:
                    queue.append(q.left)
                if q.right:
                    queue.append(q.right)
            ret.append(num)




    def pruneTree(self, root):
        if not root:
            #这个root判断树是否空
            return root
        root.left = self.pruneTree(root.left)
        root.right = self.pruneTree(root.right)
        if root.val == 0 and not root.left and not root.right:
            #处理需要剪枝的数据
            return None
        #这个是最后返回的数据
        return root



def offer():
    S = Solution()

    # result = S.twoSum([2,3,4],6)

    # result = S.threeSum([-1,0,1,2,-1,-4])

    # result = S.minSubArrayLen(7,[2,3,1,2,4,3])

    # result = S.numSubarrayProductLessThanK([10,5,2,6],100)

    # result = S.subarraySum([1,2,3],3)

    # result = S.checkInclusion("ab","eidbaooo")

    # result = S.findAnagrams("abab","ab")

    # result = S.lengthOfLongestSubstring("abcabcbb")

    # result = S.isPalindrome("A man, a plan, a canal: Panama")

    # result = S.validPalindrome("abca")

    result = S.isAnagram("anagram","nagaram")

    Log_debug.logger.info(f'result : {result}')





def qianzuihe():
    S = Solution()

    # result = S.subarraySum([2,-1,1,-4],3)
    # result = S.subarraySum([-1,2,1,-4],3)

    # result = S.pivotIndex([1, 7, 3, 6, 5, 6])

    # result = S.corpFlightBookings( [[1,2,10],[2,3,20],[2,5,25]],5)

    # result = S.largestAltitude([-5,1,5,0,-7])

    result = S.waysToPartition([22,4,-25,-20,-15,15,-16,7,19,-10,0,-13,-14],-33)

    Log_debug.logger.info(f'result : {result}')









def donggui():
    S = Solution()

    # result = S.climbStairs(4)
    # result = S.climbStairs(8)

    # result = S.rob([2,7,9,3,1])

    # result = S.rob_1([2, 7, 9, 3, 1])

    result = S.threeSumClosest([-1,2,1,-4],1)

    Log_debug.logger.info(f'result : {result}')







def erfen():
    S = Solution()
    # result = S.shipWithinDays([1,2,3,1,1],4)
    # result = S.shipWithinDays([3,2,2,4,1,4], 3)

    # result = S.searchInsert([1,3,4,6],5)
    # result = S.searchInsert([1, 3, 4, 6], 3)

    # result = S.minEatingSpeed([30,11,23,4,20],6)
    # result = S.minEatingSpeed([30,11,23,4,20], 5)

    # result = S.search([4,5,6,7,0,1,2],1)
    # result = S.search([6, 4], 4)




    Log_debug.logger.info(f'result : {result}')











def main():
    # erfen()
    # donggui()
    # qianzuihe()
    offer()






if __name__ == "__main__":
    main()