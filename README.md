# Go.Algo

> В данном репозитории будет опубликована среда в которой будут взаимодействовать агенты между собой

## Зависимости 

> Python 3.11

### Структура среды в которой будут взаимодействовать агенты

Агент может:
 - Купить тикет
 - Продать тикет
 - Ничего не делать 

Биржа может:
 - Исполнить заявку агента - если в стакане есть подходящие предложение на покупку или продажу тикета
 - Передает информацию об текущем состоянии стакана и заявках в стакане

Портфель 
> Совокупность ценных инструментов которые купил агент на бирже в результате исполнения тикета  

У портфеля есть:
- Все ценные инструменты которые купил агент во время торговой сессии
- Их текущая на данный момент цена
- Потери или доходы которые получены в результате торгов

Торговая сессия
> Может длиться от 2х часов до 8 часов(виртуального времени), за это время агенты получают информацию о движении акций 

### Формат обучения 
Агент покупает/продает тикеты на бирже в свой портфель, далее и подаем в среду следующие 5 минут и агент выбирает опять 
из 3х действий купить/продать/ничего не делать. Побеждает тот агент, который принес больше всего дохода по результатам 
своих действий, дальше гены победителей мы скрещиваем и добавляем ген мутации и все повторяется снова

## Начало обучения

0. Задается период за который у нас будут проходить обучение и их бюджет
1. `Биржа` получает информацию обо всех акциях за задачный период времени из `moexalgo`
2. `Агенты` имеют у себя бюджет для трейдинга, `ML` алгоритм ранжирует по показателям акции которые могут быть интересны в данный момент
3. `Агент` выбирает любую акцию из заданного диапазона и получает историю котировок за прошлый месяц
4. `Агент` принимает решение о покупке или продаже данной акции, путем выставление заявки в текущей цене
5. `Биржа` исполняет заявку агента и добавляет акцию в портфель инвестора
6. `Биржа` подает информацию об текущем портфеле агенту и его результатах инвестирования на следующие 5 минут
7. `Агент` снова принимает решение

> Такой цикл повторяется в течении 2/4/6/8 часов



