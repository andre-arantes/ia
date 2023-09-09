def findDecision(obj): #obj[0]: Historia do credito, obj[1]: Divida, obj[2]: Garantias, obj[3]: Renda anual
	# {"feature": "Renda anual", "instances": 14, "metric_value": 1.5306, "depth": 1}
	if obj[3] == '>35000':
		# {"feature": "Historia do credito", "instances": 7, "metric_value": 1.1488, "depth": 2}
		if obj[0] == 'Desconhecida':
			# {"feature": "Garantias", "instances": 3, "metric_value": 0.9183, "depth": 3}
			if obj[2] == 'Nenhuma':
				# {"feature": "Divida", "instances": 2, "metric_value": 1.0, "depth": 4}
				if obj[1] == 'Baixa':
					return 'Alto'
				else: return 'Alto'
			elif obj[2] == 'Adequada':
				return 'Baixo'
			else: return 'Baixo'
		elif obj[0] == 'Boa':
			return 'Baixo'
		elif obj[0] == 'Ruim':
			return 'Moderado'
		else: return 'Moderado'
	elif obj[3] == '>=15000 a <=35000':
		# {"feature": "Divida", "instances": 4, "metric_value": 1.0, "depth": 2}
		if obj[1] == 'Alta':
			# {"feature": "Historia do credito", "instances": 3, "metric_value": 0.9183, "depth": 3}
			if obj[0] == 'Desconhecida':
				return 'Alto'
			elif obj[0] == 'Boa':
				return 'Moderado'
			elif obj[0] == 'Ruim':
				return 'Alto'
			else: return 'Alto'
		elif obj[1] == 'Baixa':
			return 'Moderado'
		else: return 'Moderado'
	elif obj[3] == '<15000':
		return 'Alto'
	else: return 'Alto'