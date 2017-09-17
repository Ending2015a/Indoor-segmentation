import LogWriter as logw

print('open one file')
log = logw.LogWriter('hello.txt', True, name='hello')
log.Log('Hello')
log.Error('Hello Hello')
log.Error('Hello22')


log.clear()

print('open two files')
log2 = logw.LogWriter('hello2.txt', True, name='hello2')
log2.Log('Fuck you')
log2.Error('FFFFF')

log.Warning('sdfds')


print('open same file')
log3 = logw.LogWriter('hello.txt', True, name='hello3')
log3.Log('Fuck you, I am hello3')
log.Log('asdfdsasdfdsa')

log.close()

log3.Log('FFFFFFFF hello3')
log2.Error('GGGGGG')


log.open()

log3.Log('sdfdsdf')
log.Log('HaHaHa')
log3.Log('fdsdfd')

