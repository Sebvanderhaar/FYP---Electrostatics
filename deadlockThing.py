class item:
    def __init__(self, cost: int, tier: int, type: str, bonus: list):
        self.cost = cost
        self.tier = tier
        self.type = type
        self.bonus = self.getBonus(self)
        self.attributes = {}
        self.ability = None
    def getBonus(self):
        if self.type == "weapon":
            return [self.type, 2 + self.tier*4]
        elif self.type == "vita":
            return [self.type, 8 + self.tier*3]
        elif self.type == "spirit":
            return [self.type, self.tier*4]
        else:
            raise ValueError("Invalid type string")
        
    def addAttribute(self, key: str, value: float, unit: str):
        self.attributes[key] = {"value": value, "unit": unit}

    def displayAttributes(self):
        for key, value in self.attributes.items():
            print(f"{key}: {value}")

class ability:
    def __init__(self, type: str, cooldown: float, condition: str, duration: float):
        self.type = type #passive/active
        self.cooldown = cooldown
        self.condition = condition
        self.duration = duration
        self.attributes = {}

    def addAttribute(self, key: str, value: float, unit: str):
        self.attributes[key] = {"value": value, "unit": unit}

    def displayAttributes(self):
        for key, value in self.attributes.items():
            print(f"{key}: {value}")



def basicMagazine():
    basicMagazine = item(500, 1, "weapon")
    basicMagazine.addAttribute("ammo", 26, "%")
    basicMagazine.addAttribute("weapon damage", 15, "%")

    return basicMagazine

def closeQuarters():
    closeQuarters = item(500, 1, "weapon")
    closeQuarters.addAttribute("bullet resist", 5, "%")
    closeQuarters.ability = ability("passive", 0, "<15m", 0)
    closeQuarters.ability.addAttribute("weapon damage", 25, "%")

    return closeQuarters

def headshotBooster():
    headshotBooster = item(500, 1, "weapon")
    headshotBooster.addAttribute("fire rate", 4, "%")
    headshotBooster.addAttribute("bullet shield health", 40, "")
    headshotBooster.ability = ability("passive", 7.5, "", 0)
    headshotBooster.ability.addAttribute("headshot bonus damage", 40, "")

    return headshotBooster

def highVelocityMag():
    highVelocityMag = item(500, 1, "weapon")
    highVelocityMag.addAttribute("bullet velocity", 20, "%")
    highVelocityMag.addAttribute("weapon damage", 13, "%")
    highVelocityMag.addAttribute("bullet shield", 65, "")

    return highVelocityMag

def hollowPointWard():
    hollowPointWard = item(500, 1, "weapon")
    hollowPointWard.addAttribute("spirit shield health", 95, "")
    hollowPointWard.addAttribute("spirit power", 4, "")
    hollowPointWard.ability = ability("passive", 0, f">60% health", 0)
    hollowPointWard.ability.addAttribute("weapon damage", 22, "%")

    return hollowPointWard

def monsterRounds():
    monsterRounds = item(500, 1, "weapon")
    monsterRounds.addAttribute("health", 30, "")
    monsterRounds.addAttribute("health regen", 1, "")
    monsterRounds.addAttribute("weapon damage vs npc", 30, "%")
    monsterRounds.addAttribute("bullet resist vs npc", 25, "%")

    return monsterRounds

def rapidRounds():
    rapidRounds = item(500, 1, "weapon")
    rapidRounds.addAttribute("fire rate", 11, "%")
    rapidRounds.addAttribute("sprint speed", 1, "ms")

    return rapidRounds

def resorativeShot():
    resorativeShot = item(500, 1, "weapon")
    resorativeShot.addAttribute("bullet shield health", 90, "")
    resorativeShot.addAttribute("weapon damage", 6, "%")
    resorativeShot.ability = ability("passive", 6, "", 0)
    resorativeShot.ability.addAttribute("healing from heros, bullet", 40, "")
    resorativeShot.ability.addAttribute("healing from npc/orbs, bullet", 15, "")

    return resorativeShot

def activeReload():
    activeReload = item(1250, 2, "weapon")
    activeReload.addAttribute("weapon damage", 10, "%")
    activeReload.addAttribute("ammo", 18, "%")
    activeReload.addAttribute("health", 50, "")
    activeReload.ability = ability("passive", 18, "Reloading, hit minigame", 18)
    activeReload.ability.addAttribute("fire rate", 20, "%")
    activeReload.ability.addAttribute("bullet lifesteal", 18, "%")

    return activeReload

def berserker():
    berserker = item(1250, 2, "weapon")
    berserker.addAttribute("ammo", 4, "")
    berserker.addAttribute("bullet resist", 9, "%")
    berserker.ability = ability("passive", 0, "100 damage per stack, 10 stacks max", 10)
    berserker.ability.addAttribute("weapon damage per stack", 6, "%")

    return berserker

def kineticDash():
    kineticDash = item(1250, 2, "weapon")
    kineticDash.addAttribute("health", 100, "")
    kineticDash.addAttribute("health regen", 1.5, "")
    kineticDash.ability = ability("passive", 10.5, "dash-jump", 7)
    kineticDash.ability.addAttribute("fire rate", 20, "%")
    kineticDash.ability.addAttribute("ammo", 5, "")

def longRange():
    longRange = item(1250, 2, "weapon")
    longRange.addAttribute("reload time", 20, "%")
    longRange.addAttribute("bullet shield health", 140, "")
    longRange.addAttribute("weapon damage", 10, "%")
    longRange.ability = ability("passive", 0, ">15m", 0)
    longRange.ability.addAttribute("weapon damage", 30, "%")

    return longRange

def meleeCharge():
    meleeCharge = item(1250, 2, "weapon")
    meleeCharge.addAttribute("weapon damage", 10, "%")
    meleeCharge.addAttribute("health", 75, "")
    meleeCharge.addAttribute("heavy melee range", 40, "%")
    meleeCharge.ability = ability("passive", 16, "heavy melee and bonus ammo not full", 0)
    meleeCharge.ability.addAttribute("heavy melee damage", 20, "%")
    meleeCharge.ability.addAttribute("bonus ammo", 100, "%")

    return meleeCharge

def mysticShot():
    mysticShot = item(1250, 2, "weapon")
    mysticShot.addAttribute("weapon damage", 12, "%")
    mysticShot.addAttribute("spirit power", 4, "")
    mysticShot.ability = ability("passive", 5.75, "", 0)
    mysticShot.ability.addAttribute("bonus spirit damage", 65, "x0.8")

    return mysticShot

def

def swiftStriker():
    swiftStriker = item(1250, 2, "weapon")
    swiftStriker.addAttribute("fire rate", 22, "%")
    swiftStriker.addAttribute("ammo", 10, "%")
    swiftStriker.addAttribute("bullet resist", -5, "%")

    return swiftStriker





